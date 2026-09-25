r"""
=============================================================================================
【JSR 进阶理论组件：Phase 4-B 动态粗空间自适应谱基底数学算子库 (Changing Spectral Basis)】
=============================================================================================

一、本模块的科学研究背景与数学物理动机
---------------------------------------------------------------------------------------------
在经典的两层域分解预条件方法（Two-Level Additive Schwarz Preconditioner）中，粗网格空间通常
采用几何聚合指示向量（Block-Indicator Basis）或常数基底。这在稳态扩散或小变形问题中表现优异。
然而，当物理系统发生剧烈演化（例如：高梯度相变界面移动、塑性局部化剪切带推进、流固耦合边界大变形）时，
固定的几何粗基底可能与演化后的微分算子主导低频空间产生“空间未对准”（Subspace Misalignment），
导致全局低频长波误差无法被有效消除，从而引起 Krylov（如 CG / GMRES）迭代次数剧增。

Phase 4-B 引入了“数据驱动的自适应粗空间谱基底（Adaptive Spectral Coarse Basis）”理论：
1. 【分块代理压缩】：利用正交指示矩阵 $Q \in \mathbb{R}^{n \times p}$，将全局 $n \times n$ 的大稀疏刚度矩阵 $A$
   压缩为尺寸仅为 $p \times p$ 的对称正定代理刚度矩阵（Coarse Proxy Matrix）：
       $$C = Q^T A Q$$
   由于每个网格节点仅属于一个子域，此装配过程可在 $\mathcal{O}(\text{nnz})$ 复杂度内直接流式完成，无需将 $A$ 稠密化。

2. 【局部广义谱分解】：在小维度矩阵 $C$ 上求解广义本征值问题：
       $$C v_k = \lambda_k v_k, \quad k = 1, \dots, r \quad (r \ll p)$$
   选取对应于最小本征值（即系统能量最低、阻尼最小、衰减最慢的长波模式）的前 $r$ 个本征向量 $V \in \mathbb{R}^{p \times r}$
   构成粗空间的自适应谱基底。

3. 【投影不变性残差与按需刷新机制】：
   随着时间步演化到 $t+1$，刚度矩阵变为 $A_{t+1}$，对应的代理矩阵变为 $C_{t+1}$。我们无需在每一步都重新计算昂贵的谱分解，
   而是定义“不变子空间残差（Subspace Invariance Loss）”：
       $$\epsilon = \frac{\|C_{t+1} V - V (V^T C_{t+1} V)\|_F}{\|C_{t+1} V\|_F}$$
   只有当 $\epsilon$ 超过容忍阈值或基底存活时间（Age）达到上限时，才触发新一轮谱基底重新计算。

二、纯数学实现设计与防副作用原则 (Pure-NumPy Isolation)
---------------------------------------------------------------------------------------------
本模块是完全无状态、纯数值的数学函数库：
- 零外部依赖：纯基于 NumPy 实现，不引入 PETSc 状态或全局变量。
- 符号规范化（Sign Canonicalization）：通过主成分最大绝对值分量强制正负号朝向，彻底消除本征向量符号二义性。
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping, Sequence, Tuple

import numpy as np


def _array_sha(value: np.ndarray) -> str:
    """计算连续双精度浮点数组的 SHA-256 散列指纹，用于密码学防篡改审计。"""
    array = np.ascontiguousarray(value, dtype=np.float64)
    return sha256(array.view(np.uint8)).hexdigest()


def normalized_indicator_basis(owner: Sequence[int], domain_sizes: Sequence[int],
                               factor_count: int = 144) -> np.ndarray:
    r"""
    【构建列正交分块指示基矩阵 Q (Normalized Indicator Basis)】
    -----------------------------------------------------------------------------------------
    数学定义：
      设全局未知量自由度总数为 $n$，子域（因子）总数为 $p$（默认 144）。
      如果第 $i$ 个自由度归属于第 $b$ 个子域（即 $\text{owner}[i] = b$），则：
          $$Q_{i, b} = \frac{1}{\sqrt{|\mathcal{D}_b|}}$$
      其余位置全为 0。

    性质保证：
      $Q^T Q = I_p$（列向量严格正交且单位化，满足等距嵌入性质）。
    """
    owner_array = np.asarray(owner, dtype=np.int64)
    sizes = np.asarray(domain_sizes, dtype=np.int64)
    if owner_array.ndim != 1 or sizes.shape != (int(factor_count),):
        raise ValueError("owner/domain-size shape mismatch")
    if np.any(owner_array < 0) or np.any(owner_array >= int(factor_count)):
        raise ValueError("owner contains an invalid block")
    counts = np.bincount(owner_array, minlength=int(factor_count))
    if not np.array_equal(counts, sizes) or np.any(sizes <= 0):
        raise ValueError("domain sizes do not match owner")

    # 分配密集稀疏映射矩阵 Q，尺寸为 (n, p)
    q = np.zeros((owner_array.size, int(factor_count)), dtype=np.float64)
    q[np.arange(owner_array.size), owner_array] = 1.0 / np.sqrt(
        sizes[owner_array].astype(np.float64))
    return q


def assemble_block_proxy(matrix: Mapping[str, np.ndarray], owner: Sequence[int],
                         domain_sizes: Sequence[int], factor_count: int = 144) -> np.ndarray:
    r"""
    【直接流式装配分块代理粗算子 C = Q^T * A * Q】
    -----------------------------------------------------------------------------------------
    工程挑战：
      全局矩阵 $A$ 维度极大（如数万甚至数百万），绝不可将 $A$ 转为稠密矩阵再做大矩阵乘法。

    数学化简与流式计算：
      由于 $Q$ 的每一行仅有一个非零元（即每个节点拥有唯一的所属子域）：
          $$C_{b_1, b_2} = \sum_{(i, j) \in A, \, \text{owner}[i]=b_1, \, \text{owner}[j]=b_2} \frac{A_{i, j}}{\sqrt{|\mathcal{D}_{b_1}| \cdot |\mathcal{D}_{b_2}|}}$$
      因此，只需线性遍历全局稀疏矩阵 $A$ 的 CSR 格式（非零元），按照节点所属子域归并累加至 $p \times p$ 的小矩阵 $C$ 即可。
      时间复杂度严格为 $\mathcal{O}(\text{nnz}(A))$，空间开销仅为小矩阵 $p \times p$（约 144×144 = 20736 个浮点数）。
    """
    owner_array = np.asarray(owner, dtype=np.int64)
    sizes = np.asarray(domain_sizes, dtype=np.int64)
    n = int(matrix["shape"][0])
    if tuple(int(x) for x in matrix["shape"]) != (n, n):
        raise ValueError("matrix must be square")
    if owner_array.shape != (n,) or sizes.shape != (int(factor_count),):
        raise ValueError("proxy dimensions mismatch")
    if np.any(owner_array < 0) or np.any(owner_array >= int(factor_count)):
        raise ValueError("owner out of range")
    counts = np.bincount(owner_array, minlength=int(factor_count))
    if not np.array_equal(counts, sizes) or np.any(sizes <= 0):
        raise ValueError("domain sizes do not match owner")

    # 预先计算缩放因子 1 / sqrt(|D_b|)
    scale = 1.0 / np.sqrt(sizes.astype(np.float64))
    coarse = np.zeros((int(factor_count), int(factor_count)), dtype=np.float64)
    indptr = np.asarray(matrix["indptr"], dtype=np.int64)
    indices = np.asarray(matrix["indices"], dtype=np.int64)
    data = np.asarray(matrix["data"], dtype=np.float64)
    if indptr.shape != (n + 1,) or indices.shape != data.shape:
        raise ValueError("invalid CSR arrays")

    # 遍历 CSR 非零元，流式归并到分块粗网格代理中
    for row in range(n):
        row_block = int(owner_array[row])
        lo, hi = int(indptr[row]), int(indptr[row + 1])
        for col, value in zip(indices[lo:hi], data[lo:hi]):
            col_block = int(owner_array[int(col)])
            coarse[row_block, col_block] += float(value) * scale[row_block] * scale[col_block]
    return np.ascontiguousarray(coarse, dtype=np.float64)


def _canonicalize_signs(vectors: np.ndarray) -> np.ndarray:
    r"""
    【特征向量正负号规范化 (Sign Canonicalization)】
    -----------------------------------------------------------------------------------------
    本征值分解 $C v = \lambda v$ 在数学上具有符号二义性：如果 $v$ 是特征向量，$-v$ 同样是特征向量。
    底层线性代数库（如 LAPACK / MKL）在不同硬件架构或多线程调度下，特征向量的正负符号可能出现随机翻转。
    这种翻转虽不影响子空间所张成的几何空间，但会破坏数值实验在逐 bit 级别上的可重复性（Provenance Digest）。

    规则：
      寻找每个特征向量中绝对值最大的分量（主导分量 Pivot），若该分量小于 0，则将整列乘以 -1，
      确保主导分量始终为正。
    """
    output = np.ascontiguousarray(vectors, dtype=np.float64).copy()
    for column in range(output.shape[1]):
        vector = output[:, column]
        magnitude = np.abs(vector)
        pivot = int(np.argmax(magnitude))
        if vector[pivot] < 0.0:
            output[:, column] *= -1.0
    return output


@dataclass(frozen=True)
class SpectralBasis:
    """
    【自适应谱基底不可变状态容器】
    -----------------------------------------------------------------------------------------
    :param vectors: 正交特征向量矩阵 V (尺寸 p x rank)
    :param eigenvalues: 对应的最小前 rank 个本征值 lambda
    :param build_state: 构筑该基底时所处的物理时间步编号
    :param age: 基底未被刷新的连续物理步数（存活年龄）
    :param proxy_digest: 当时所依据的代理矩阵 C 的 SHA-256 散列指纹
    """
    vectors: np.ndarray
    eigenvalues: np.ndarray
    build_state: int
    age: int
    proxy_digest: str

    @property
    def rank(self) -> int:
        """谱粗空间的降维阶数（模态数目）。"""
        return int(self.vectors.shape[1])

    @property
    def digest(self) -> str:
        """基底矩阵的唯一密码学指纹。"""
        return _array_sha(self.vectors)


def build_spectral_basis(proxy: np.ndarray, rank: int, build_state: int,
                         spd_relative_tolerance: float = 1.0e-12) -> SpectralBasis:
    """
    【构建确定性低频谱基底 (Lowest-Eigenmode Spectral Basis)】
    -----------------------------------------------------------------------------------------
    算法流程：
      1. 严格检查小矩阵 $C$ 的对称性与正定性（SPD 性质）；
      2. 调用 `np.linalg.eigh` 求解标准对称本征值问题；
      3. 截取最小的前 `rank` 个本征值及对应本征向量；
      4. 调用 `_canonicalize_signs` 进行符号规范化；
      5. 严格验证正交单位性：$V^T V \approx I$；
      6. 返回封装好的不可变 `SpectralBasis` 对象。
    """
    c = np.asarray(proxy, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or not np.all(np.isfinite(c)):
        raise ValueError("proxy must be a finite square matrix")
    if not (1 <= int(rank) <= c.shape[0]):
        raise ValueError("spectral rank is out of range")

    # 对称化处理以消除浮点舍入微小非对称误差
    symmetry_error = float(np.max(np.abs(c - c.T)))
    symmetric = 0.5 * (c + c.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0e-300)

    # 正定性检测：最小本征值必须大于容差
    if float(eigenvalues[0]) <= float(spd_relative_tolerance) * scale:
        raise np.linalg.LinAlgError("coarse proxy is not positive definite")
    if symmetry_error > 1.0e-10 * max(float(np.max(np.abs(symmetric))), 1.0):
        raise ValueError("coarse proxy is not sufficiently symmetric")

    # 截取前 rank 个最小能量本征模式
    selected_values = np.ascontiguousarray(eigenvalues[:int(rank)], dtype=np.float64)
    selected_vectors = _canonicalize_signs(eigenvectors[:, :int(rank)])

    # 正交性断言检查
    if np.max(np.abs(selected_vectors.T.dot(selected_vectors) -
                np.eye(int(rank)))) > 1.0e-10:
        raise ValueError("spectral basis is not orthonormal")

    return SpectralBasis(
        vectors=selected_vectors,
        eigenvalues=selected_values,
        build_state=int(build_state),
        age=0,
        proxy_digest=_array_sha(symmetric),
    )


def projection_residual(proxy: np.ndarray, basis_vectors: np.ndarray) -> float:
    r"""
    【度量子空间不变性退化损失 (Subspace Invariance Residual)】
    -----------------------------------------------------------------------------------------
    物理内涵：
      若基底 $V$ 依然是当前代理刚度矩阵 $C$ 的精确不变子空间，则 $C V$ 应该完全落在 $V$ 张成的空间内，
      即正交投影算子 $P_V = V V^T$ 作用于 $C V$ 后应有：
          $$P_V (C V) = V (V^T C V) = C V$$
      两者之差的 Frobenius 范数即为子空间漂移残差：
          $$R = C V - V (V^T C V)$$
      相对残差公式：
          $$\epsilon = \frac{\|R\|_F}{\|C V\|_F}$$
      若 $\epsilon \approx 0$，说明旧基底依然能够完美捕捉当前物理系统的主要低频模态；
      若 $\epsilon \gg 0$，说明物理界面移动导致旧模态严重失真，必须刷新。
    """
    c = np.asarray(proxy, dtype=np.float64)
    v = np.asarray(basis_vectors, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or v.ndim != 2 or v.shape[0] != c.shape[0]:
        raise ValueError("proxy/basis dimensions mismatch")
    cv = c.dot(v)
    residual = cv - v.dot(v.T.dot(cv))
    return float(np.linalg.norm(residual, ord="fro") /
                 max(float(np.linalg.norm(cv, ord="fro")), 1.0e-300))


def should_refresh_basis(proxy: np.ndarray, basis: SpectralBasis,
                         epsilon_threshold: float = 0.05,
                         max_age_before_refresh: int = 2) -> Tuple[bool, float, str]:
    r"""
    【双指标驱动的因果刷新决策门限 (Basis Refresh Policy)】
    -----------------------------------------------------------------------------------------
    本函数严格执行冻结协议中规定的无未来信息的因果规则：
      1. 动力学退化触发：若子空间残差 $\epsilon > 0.05$（即空间失真超过 5%），判定为必须刷新；
      2. 周期保底触发：若基底存活时间 $\text{age} \ge 2$ 个步长，强制刷新以防误差累积；
      3. 否则保留旧基底（Retain），零开销进入下一步。
    :return: (是否刷新, 当前残差值, 触发原因)
    """
    epsilon = projection_residual(proxy, basis.vectors)
    if epsilon > float(epsilon_threshold):
        return True, epsilon, "projection_residual"
    if int(basis.age) >= int(max_age_before_refresh):
        return True, epsilon, "basis_age"
    return False, epsilon, "retain"


def reduced_coarse_operator(proxy: np.ndarray, basis_vectors: np.ndarray) -> np.ndarray:
    r"""
    【构建低维压缩粗网格刚度矩阵 E = V^T * C * V】
    -----------------------------------------------------------------------------------------
    由于 $V \in \mathbb{R}^{p \times r}$（如 $144 \times 32$），压缩后的粗算子 $E$ 仅为 $32 \times 32$ 的对称正定方阵。
    在求解两层预条件子时，仅需对极小尺寸的 $E$ 进行直接求逆或 Cholesky 分解，计算耗时微秒级。
    """
    c = np.asarray(proxy, dtype=np.float64)
    v = np.asarray(basis_vectors, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or v.ndim != 2 or v.shape[0] != c.shape[0]:
        raise ValueError("proxy/basis dimensions mismatch")
    # 强制对称化以消除浮点非对称误差
    vt_c_v = v.T.dot(c).dot(v)
    return np.ascontiguousarray(0.5 * (vt_c_v + vt_c_v.T), dtype=np.float64)


__all__ = [
    "SpectralBasis",
    "assemble_block_proxy",
    "build_spectral_basis",
    "normalized_indicator_basis",
    "projection_residual",
    "reduced_coarse_operator",
    "should_refresh_basis",
]

