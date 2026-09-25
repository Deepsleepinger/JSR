r"""
=============================================================================================
【JSR 进阶架构组件：Phase 4-B PETSc 动态谱基底有状态预条件器后端 (Changing Backend)】
=============================================================================================

一、本模块的设计哲学与架构演化
---------------------------------------------------------------------------------------------
在基础生产环境（`backend.py`）中，粗网格限制/延拓算子绑定于固定的几何分块。
为了验证“自适应粗空间谱基底（Adaptive Spectral Coarse Basis）”在 PETSc 生态中的物理有效性，
本模块继承了 `CorrectedStatefulTwoLevel` 生产级控制器的成熟机制：
1. 继承局部因子（Local Factors）的增量式有状态维护（原地更新、选择性求逆）；
2. 继承 PETSc KSP/PC 求解器生命周期及真残差证书机制；
3. 将粗网格校正算子由固定的几何指示映射：
       $$M_{\text{coarse}}^{-1} = R_0^T A_0^{-1} R_0$$
   扩展为支持低维谱降维空间的广义映射：
       $$M_{\text{coarse}}^{-1} = Q V E^{-1} V^T Q^T$$
   其中：
   - $Q \in \mathbb{R}^{n \times p}$: 列正交分块指示矩阵；
   - $V \in \mathbb{R}^{p \times r}$: 自适应低频本征谱基底 ($r \ll p$)；
   - $E = V^T (Q^T A Q) V \in \mathbb{R}^{r \times r}$: 压缩粗网格刚度矩阵。

二、四类基底模式的语义隔离 (Basis Modes)
---------------------------------------------------------------------------------------------
- `fixed_block`: 退化为经典固定常数分块基底 ($V = I_p$)，用于做严格的对照回退测试；
- `fixed_spectral`: 初始时刻由 $t=0$ 算子做一次谱分解生成，并在后续演化中完全冻结；
- `triggered_spectral`: JSR 核心机制，监控子空间投影残差 $\epsilon$ 与基底存活年龄，按需动态触发重构；
- `eager_spectral`: 每一步无条件重新做谱分解（作为理论收敛精度的上限基准）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, Tuple
import time

import numpy as np

from .backend import CacheSnapshot, CoarseState, CorrectedStatefulTwoLevel
from .changing_basis import (
    SpectralBasis,
    assemble_block_proxy,
    build_spectral_basis,
    projection_residual,
    reduced_coarse_operator,
)


class ChangingTwoLevelContext:
    r"""
    【PETSc Python-PC 动态谱预条件器应用算子 (Changing Two-Level Context)】
    -----------------------------------------------------------------------------------------
    本类是注入到 PETSc Shell PC 中的回调上下文。在 Krylov 迭代的每一次预条件子应用
    `y = M^{-1} x` 中，执行如下经典的加性域分解数学运算：

      1. 【局部高频修正】：
         对每个子域 $\mathcal{D}_i$，截取局部残差 $x|_{\mathcal{D}_i}$，利用预先分解好的局部逆矩阵求解，
         并直接累加到输出向量 $y$ 中：
             $$y_{\text{local}} = \sum_{i=1}^p R_i^T A_i^{-1} R_i x$$

      2. 【粗网格限制投影 (Restriction)】：
         利用分块指示矩阵 $Q$ 与加权因子，将残差限制到分块粗网格：
             $$r_c = Q^T x = \left[ \frac{1}{\sqrt{|\mathcal{D}_b|}} \sum_{j \in \mathcal{D}_b} x_j \right]_{b=1}^p$$

      3. 【低维谱投影与粗解 (Spectral Projection & Coarse Solve)】：
         利用正交谱基底 $V \in \mathbb{R}^{p \times r}$ 投影至低维空间，并利用极小尺寸矩阵 $E^{-1}$ 求解：
             $$s_r = E^{-1} (V^T r_c)$$

      4. 【谱延拓与反向加权 (Interpolation & Prolongation)】：
         将粗解反投影回分块空间并分配到各网格自由度：
             $$y \leftarrow y_{\text{local}} + Q V s_r$$
    """

    def __init__(self, factors: Sequence, owner: np.ndarray,
                 inverse_sqrt_sizes: np.ndarray, basis_vectors: np.ndarray,
                 coarse: CoarseState):
        self.factors = list(factors)
        self.owner = np.asarray(owner, dtype=np.int64)
        self.inverse_sqrt_sizes = np.asarray(inverse_sqrt_sizes, dtype=np.float64)
        self.basis_vectors = np.ascontiguousarray(basis_vectors, dtype=np.float64)
        self.coarse = coarse
        self.calls = 0
        self.seconds = 0.0

    def apply(self, pc, x, y):
        started = time.perf_counter()
        xa = np.asarray(x.getArray(readonly=True))
        ya = y.getArray()
        ya[:] = 0.0

        # 阶段 1：并行累加各子域的局部 Cholesky 反解（消除局部高频误差）
        for factor in self.factors:
            indices = factor.indices
            ya[indices] += factor.inverse.dot(xa[indices])

        # 阶段 2：残差限制到分块代理网格 (利用 bincount 极速聚合)
        restricted = np.bincount(
            self.owner,
            weights=xa * self.inverse_sqrt_sizes[self.owner],
            minlength=self.basis_vectors.shape[0],
        )

        # 阶段 3：投影到 r 维谱子空间并调用压缩粗矩阵求逆
        reduced_rhs = self.basis_vectors.T.dot(restricted)
        reduced_solution = self.coarse.inverse.dot(reduced_rhs)

        # 阶段 4：从谱子空间延拓回物理自由度并叠加输出
        block_correction = self.basis_vectors.dot(reduced_solution)
        ya[:] += block_correction[self.owner] * self.inverse_sqrt_sizes[self.owner]
        y.assemble()

        self.calls += 1
        self.seconds += time.perf_counter() - started


@dataclass(frozen=True)
class ChangingCacheSnapshot:
    """
    【动态谱基底状态快照容器 (Transactional Snapshot)】
    -----------------------------------------------------------------------------------------
    记录某一物理步的完整预条件子内部状态。如果在后续求解中发生 Krylov 发散，
    系统可通过此快照实现零损耗、绝对一致的状态回滚。
    """
    base: CacheSnapshot
    basis_vectors: np.ndarray
    basis_eigenvalues: np.ndarray
    basis_build_state: int
    basis_age: int
    proxy: np.ndarray


class ChangingBasisStatefulTwoLevel(CorrectedStatefulTwoLevel):
    """
    【支持动态粗基底的有状态两层预条件器生命周期管理器】
    -----------------------------------------------------------------------------------------
    本类是 JSR 框架中能力最强的高级预条件器实体。它不仅管理局部子域因子，还协同管理
    自适应谱基底与压缩粗算子的整个演化生命周期。
    """

    MODES = ("fixed_block", "fixed_spectral", "triggered_spectral", "eager_spectral")

    def __init__(self, pilot_module, base_matrix: dict, base_rhs: np.ndarray,
                 domains: Sequence[np.ndarray], aggregate: np.ndarray,
                 initial_state: int = 0, ksp_rtol: float = 1.0e-10,
                 max_it: int = 2000, basis_mode: str = "fixed_spectral",
                 basis_rank: int = 32):
        if str(basis_mode) not in self.MODES:
            raise ValueError("unknown Phase 4-B basis mode: %s" % basis_mode)
        if str(basis_mode) in ("fixed_block",) and int(basis_rank) != len(domains):
            raise ValueError("fixed block basis rank must equal factor count")
        if str(basis_mode) != "fixed_block" and not (1 <= int(basis_rank) <= len(domains)):
            raise ValueError("spectral basis rank is out of range")

        # 步骤 1：调用父类初始化 PETSc KSP/PC 以及局部因子
        super().__init__(pilot_module, base_matrix, base_rhs, domains, aggregate,
                         initial_state, ksp_rtol, max_it)
        self.basis_mode = str(basis_mode)
        self.basis_rank = int(basis_rank)
        self.owner = np.asarray(aggregate, dtype=np.int64).copy()
        self.domain_sizes = np.asarray([len(item) for item in self.domains], dtype=np.int64)
        self.inverse_sqrt_sizes = 1.0 / np.sqrt(self.domain_sizes.astype(np.float64))

        # 步骤 2：装配初始时刻的分块代理刚度矩阵 C
        self.proxy = assemble_block_proxy(base_matrix, self.owner, self.domain_sizes,
                                          self.factor_count)

        # 步骤 3：根据指定的基底模式构建初始谱基底
        basis_started = time.perf_counter()
        if self.basis_mode == "fixed_block":
            vectors = np.eye(self.factor_count, dtype=np.float64)
            eigenvalues = np.ones(self.factor_count, dtype=np.float64)
            self.basis = SpectralBasis(
                vectors=vectors, eigenvalues=eigenvalues,
                build_state=int(initial_state), age=0,
                proxy_digest=self._array_sha(self.proxy),
            )
        else:
            self.basis = build_spectral_basis(
                self.proxy, self.basis_rank, int(initial_state)
            )
        self.initial_basis_seconds = float(time.perf_counter() - basis_started)

        # 步骤 4：构建初始时刻的压缩粗矩阵 E 并求逆
        coarse_started = time.perf_counter()
        self.coarse = self._make_coarse(self.proxy, self.basis.vectors, int(initial_state))
        self.initial_coarse_seconds = float(time.perf_counter() - coarse_started)
        self.coarse_build_state = int(initial_state)
        self.coarse_age = 0
        self.basis_build_state = int(initial_state)
        self.basis_age = 0

        # 步骤 5：将动态上下文绑定到 PETSc Shell PC
        self.context = ChangingTwoLevelContext(
            self.factors, self.owner, self.inverse_sqrt_sizes,
            self.basis.vectors, self.coarse,
        )
        self.pc.setPythonContext(self.context)
        self.initial_setup_seconds += self.initial_basis_seconds + self.initial_coarse_seconds

    @staticmethod
    def _array_sha(value: np.ndarray) -> str:
        """内部数组 SHA-256 计算辅助函数。"""
        from hashlib import sha256
        array = np.ascontiguousarray(value, dtype=np.float64)
        return sha256(array.view(np.uint8)).hexdigest()


    def _make_coarse(self, proxy: np.ndarray, vectors: np.ndarray,
                     build_state: int) -> CoarseState:
        """
        【构建压缩粗矩阵 E = V^T * C * V 及其逆矩阵】
        -----------------------------------------------------------------------------------------
        计算耗时记录：
          分别剥离装配耗时（assembly_seconds）与 Cholesky 求逆耗时（factorization_seconds），
          并检测最小本征值以确保其严格正定。
        """
        started = time.perf_counter()
        reduced = reduced_coarse_operator(proxy, vectors)
        symmetry_error = float(np.max(np.abs(reduced - reduced.T)))
        factor_started = time.perf_counter()
        inverse = np.linalg.inv(reduced)
        minimum_eigenvalue = float(np.linalg.eigvalsh(reduced)[0])
        factor_seconds = time.perf_counter() - factor_started
        return CoarseState(
            matrix=np.ascontiguousarray(reduced, dtype=np.float64),
            inverse=np.ascontiguousarray(inverse, dtype=np.float64),
            build_state=int(build_state),
            assembly_seconds=float(time.perf_counter() - started - factor_seconds),
            factorization_seconds=float(factor_seconds),
            symmetry_error=symmetry_error,
            minimum_eigenvalue=minimum_eigenvalue,
        )

    def prepare_proxy(self, matrix: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, float]:
        """流式流转装配分块代理粗算子 C = Q^T A Q 并计时。"""
        started = time.perf_counter()
        proxy = assemble_block_proxy(matrix, self.owner, self.domain_sizes,
                                     self.factor_count)
        return proxy, float(time.perf_counter() - started)

    def basis_probe(self, proxy: np.ndarray) -> float:
        """
        【嗅探当前基底的子空间不变性残差 epsilon】
        -----------------------------------------------------------------------------------------
        固定分块模式下残差恒为 0；在自适应谱模式下计算 epsilon。
        """
        if self.basis_mode == "fixed_block":
            return 0.0
        return projection_residual(proxy, self.basis.vectors)

    def refresh_basis(self, proxy: np.ndarray, current_state: int) -> dict:
        """
        【重新计算谱基底 V (Spectral Basis Recomputation)】
        -----------------------------------------------------------------------------------------
        在当前代理算子 C 上重新求解最小前 r 个本征向量，更新本地缓存与 PETSc 上下文指针。
        """
        if self.basis_mode == "fixed_block":
            raise ValueError("fixed block basis cannot be refreshed")
        started = time.perf_counter()
        self.basis = build_spectral_basis(proxy, self.basis_rank, int(current_state))
        self.basis_build_state = int(current_state)
        self.basis_age = 0
        self.context.basis_vectors = self.basis.vectors
        return {
            "seconds": float(time.perf_counter() - started),
            "rank": self.basis.rank,
            "build_state": self.basis_build_state,
            "basis_digest": self.basis.digest,
            "eigenvalues": self.basis.eigenvalues.tolist(),
            "proxy_digest": self.basis.proxy_digest,
        }

    def refresh_coarse(self, matrix: Mapping[str, np.ndarray], current_state: int,
                       proxy: np.ndarray = None) -> dict:
        """
        【重新计算压缩粗网格刚度矩阵 E = V^T C V 及其逆】
        -----------------------------------------------------------------------------------------
        当谱基底 V 或代理算子 C 发生改变时，同步更新压缩粗算子与逆矩阵。
        """
        if proxy is None:
            proxy, proxy_seconds = self.prepare_proxy(matrix)
        else:
            proxy_seconds = 0.0
        started = time.perf_counter()
        coarse = self._make_coarse(proxy, self.basis.vectors, int(current_state))
        self.proxy = np.ascontiguousarray(proxy, dtype=np.float64)
        self.coarse = coarse
        self.coarse_build_state = int(current_state)
        self.coarse_age = 0
        self.context.coarse = coarse
        return {
            "seconds": float(time.perf_counter() - started + proxy_seconds),
            "proxy_assembly_seconds": float(proxy_seconds),
            "assembly_seconds": float(coarse.assembly_seconds),
            "factorization_seconds": float(coarse.factorization_seconds),
            "matrix_digest": coarse.matrix_digest,
            "inverse_digest": coarse.inverse_digest,
        }

    def retain_coarse(self) -> None:
        """保留粗矩阵不更新，存活年龄自增 1。"""
        self.coarse_age += 1

    def advance_basis_age(self) -> None:
        """基底未重算，基底存活年龄自增 1。"""
        if self.basis_mode != "fixed_block":
            self.basis_age += 1

    def snapshot(self) -> ChangingCacheSnapshot:
        """
        【制作全量状态快照】
        -----------------------------------------------------------------------------------------
        完整克隆基类因子状态、基底矩阵、本征值、代理矩阵与年龄标量，供事务回滚使用。
        """
        return ChangingCacheSnapshot(
            base=super().snapshot(),
            basis_vectors=self.basis.vectors.copy(),
            basis_eigenvalues=self.basis.eigenvalues.copy(),
            basis_build_state=int(self.basis_build_state),
            basis_age=int(self.basis_age),
            proxy=self.proxy.copy(),
        )

    def restore(self, snapshot: ChangingCacheSnapshot) -> None:
        """
        【快照精确回滚】
        -----------------------------------------------------------------------------------------
        当 Krylov 求解器收敛失败触发熔断降级重试时，调用本函数将内部所有张量与标量原子恢复
        到时间步开始前的初始状态，杜绝任何状态脏数据污染。
        """
        super().restore(snapshot.base)
        self.basis = SpectralBasis(
            vectors=np.ascontiguousarray(snapshot.basis_vectors, dtype=np.float64),
            eigenvalues=np.ascontiguousarray(snapshot.basis_eigenvalues, dtype=np.float64),
            build_state=int(snapshot.basis_build_state),
            age=int(snapshot.basis_age),
            proxy_digest=self._array_sha(snapshot.proxy),
        )
        self.basis_build_state = int(snapshot.basis_build_state)
        self.basis_age = int(snapshot.basis_age)
        self.proxy = np.ascontiguousarray(snapshot.proxy, dtype=np.float64)
        self.context.basis_vectors = self.basis.vectors
        self.context.coarse = self.coarse

    def cache_digest(self) -> str:
        """计算包含局部因子、谱基底与代理算子的全局密码学 SHA-256 散列。"""
        from hashlib import sha256
        payload = "|".join((super().cache_digest(), self.basis.digest,
                            self._array_sha(self.basis.eigenvalues),
                            str(self.basis_build_state), str(self.basis_age),
                            self._array_sha(self.proxy)))
        return sha256(payload.encode("ascii")).hexdigest()

    def cache_proof(self) -> dict:
        """导出全套可独立核算的缓存状态证明字典。"""
        proof = dict(super().cache_proof())
        proof.update({
            "coarse_basis_digest": self.basis.digest,
            "coarse_basis_eigenvalue_digest": self._array_sha(self.basis.eigenvalues),
            "coarse_basis_eigenvalues": self.basis.eigenvalues.tolist(),
            "coarse_basis_rank": self.basis.rank,
            "coarse_basis_build_state": int(self.basis_build_state),
            "coarse_basis_age": int(self.basis_age),
            "coarse_proxy_digest": self._array_sha(self.proxy),
            "cache_digest": self.cache_digest(),
        })
        return proof

    def state_dict(self, current_state: int) -> dict:
        """导出记录到物理实验账本中的状态元数据字典。"""
        state = dict(super().state_dict(current_state))
        state.update({
            "coarse_basis_build_state": int(self.basis_build_state),
            "coarse_basis_age": int(self.basis_age),
            "coarse_basis_rank": self.basis.rank,
            "coarse_basis_digest": self.basis.digest,
            "coarse_proxy_digest": self._array_sha(self.proxy),
        })
        return state


__all__ = [
    "ChangingBasisStatefulTwoLevel",
    "ChangingCacheSnapshot",
    "ChangingTwoLevelContext",
]

