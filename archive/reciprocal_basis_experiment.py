import os
import math
import heapq
from collections import Counter

import numpy as np

P = 257
N = P - 1


def signed_inverse_reciprocal(x: int, p: int = P) -> float:
    """Map x in F_p^* to 1/r, where r is the centered integer representative
    of x^{-1} mod p. Example for p=17: inverse 12 is represented as -5,
    so the mapped value is -1/5.
    """
    r = pow(int(x), -1, p)
    if r > (p - 1) // 2:
        r -= p
    return 1.0 / r


def build_reciprocal_matrix() -> np.ndarray:
    M = np.empty((N, N), dtype=np.float64)
    for row, a in enumerate(range(1, P)):
        x = 1
        for n in range(N):
            M[row, n] = signed_inverse_reciprocal(x)
            x = (x * a) % P
    return M


def polar_orthogonalize(M: np.ndarray) -> np.ndarray:
    """Nearest orthogonal matrix to M in Frobenius norm: Q = U V^T."""
    U, _, VT = np.linalg.svd(M, full_matrices=False)
    return U @ VT


def build_real_dft() -> np.ndarray:
    n = np.arange(N)
    rows = [np.ones(N) / np.sqrt(N)]
    for k in range(1, N // 2):
        rows.append(np.sqrt(2 / N) * np.cos(2 * np.pi * k * n / N))
        rows.append(np.sqrt(2 / N) * np.sin(2 * np.pi * k * n / N))
    rows.append(((-1.0) ** n) / np.sqrt(N))
    return np.array(rows)


def build_dct2() -> np.ndarray:
    n = np.arange(N)
    D = np.empty((N, N), dtype=np.float64)
    D[0] = 1 / np.sqrt(N)
    for k in range(1, N):
        D[k] = np.sqrt(2 / N) * np.cos(np.pi * (n + 0.5) * k / N)
    return D


def huffman_bits(values) -> int:
    counts = Counter(map(int, values))
    heap = list(counts.values())
    if len(heap) <= 1:
        return 0
    heapq.heapify(heap)
    total = 0
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        s = a + b
        total += s
        heapq.heappush(heap, s)
    return total


def full_blocks_from_file(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = f.read()
    count = len(data) // N
    if count == 0:
        return np.empty((0, N), dtype=np.float64)
    return np.frombuffer(data[:count * N], dtype=np.uint8).astype(np.float64).reshape(count, N)


def transform_metrics(X: np.ndarray, T: np.ndarray):
    # Remove each block's mean so DC does not dominate the comparison.
    Xc = X - X.mean(axis=1, keepdims=True)
    C = Xc @ T.T
    energy = C * C
    total = energy.sum(axis=1)
    sorted_energy = np.sort(energy, axis=1)[:, ::-1]
    cumulative = np.cumsum(sorted_energy, axis=1)

    out = {}
    for k in (1, 4, 8, 16, 32, 64):
        frac = np.ones(len(X), dtype=float)
        nz = total > 1e-12
        frac[nz] = cumulative[nz, k - 1] / total[nz]
        out[f"E{k}"] = float(np.mean(frac))

    # Rounding is only a transform-coding proxy, not a lossless codec.
    Cq = np.rint(C).astype(np.int64)
    out["zero_fraction"] = float(np.mean(Cq == 0))
    out["huffman_bits_per_coeff"] = huffman_bits(Cq.ravel()) / Cq.size

    # Because T is orthogonal, inverse is T. This measures the distortion caused
    # solely by rounding transformed coefficients to the nearest integer.
    Xr = Cq @ T
    out["roundtrip_rmse_after_rounding"] = float(np.sqrt(np.mean((Xr - Xc) ** 2)))
    return out


def synthetic_energy(x: np.ndarray, T: np.ndarray):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    c = T @ x
    e = np.sort(c * c)[::-1]
    total = e.sum()
    return {f"E{k}": float(e[:k].sum() / total) for k in (1, 4, 8, 16, 32, 64)}


if __name__ == "__main__":
    M = build_reciprocal_matrix()
    Q = polar_orthogonalize(M)
    F = build_real_dft()
    D = build_dct2()

    s = np.linalg.svd(M, compute_uv=False)
    print("=== Matrix diagnostics ===")
    print("rank(M):", np.linalg.matrix_rank(M))
    print("condition number(M):", s[0] / s[-1])
    print("max |Q Q^T - I|:", np.max(np.abs(Q @ Q.T - np.eye(N))))

    Mnorm = M / np.linalg.norm(M, axis=1, keepdims=True)
    row_cos = np.sum(Q * Mnorm, axis=1)
    print("row cosine similarity M -> Q: min/median/max =",
          float(row_cos.min()), float(np.median(row_cos)), float(row_cos.max()))

    corr_to_dft = np.abs(Q @ F.T).max(axis=1)
    print("max correlation of each Q row to any real-DFT row: median =",
          float(np.median(corr_to_dft)))

    transforms = {"Custom-Q": Q, "Real-DFT": F, "DCT-II": D}

    # Synthetic sanity checks.
    n = np.arange(N)
    synthetic = {
        "original reciprocal row a=3": M[2],
        "3-sparse Custom-Q mixture": 3 * Q[10] + 2 * Q[20] - 1.5 * Q[33],
        "sine mixture": np.sin(2*np.pi*7*n/N) + 0.4*np.cos(2*np.pi*13*n/N),
        "period-8 waveform": np.tile(np.array([1, -1, 1, -1, 0.5, -0.5, 0.2, -0.2]), 32),
    }

    print("\n=== Synthetic energy compaction ===")
    for sname, x in synthetic.items():
        print("\n", sname)
        for tname, T in transforms.items():
            m = synthetic_energy(x, T)
            print(f"  {tname:10s} E1={m['E1']:.4f} E4={m['E4']:.4f} E16={m['E16']:.4f} E64={m['E64']:.4f}")

    # Real files available in the working directory/container.
    paths = [
        'Wildlife.wmv',
    ]

    print("\n=== Real-file byte tests ===")
    for path in paths:
        if not os.path.exists(path):
            continue
        X = full_blocks_from_file(path)
        print(f"\n{os.path.basename(path)}: {len(X)} full blocks")
        for tname, T in transforms.items():
            m = transform_metrics(X, T)
            print(
                f"  {tname:10s} E16={m['E16']:.4f} E64={m['E64']:.4f} "
                f"Huff={m['huffman_bits_per_coeff']:.4f} bits/coeff "
                f"zero={m['zero_fraction']:.4f} RMSE={m['roundtrip_rmse_after_rounding']:.4f}"
            )

    # Decode PNGs to raw grayscale pixels as a more meaningful uncompressed-signal test.
    try:
        from PIL import Image
        print("\n=== Decoded grayscale image tests ===")
        for path in [
            "/mnt/data/image(20260930-034720).png",
            "/mnt/data/1b922f7f-d4ea-46e3-898c-71fac17789d1.png",
        ]:
            if not os.path.exists(path):
                continue
            im = Image.open(path).convert("L")
            raw = np.asarray(im, dtype=np.uint8).ravel().tobytes()
            count = len(raw) // N
            X = np.frombuffer(raw[:count * N], dtype=np.uint8).astype(np.float64).reshape(count, N)
            print(f"\n{os.path.basename(path)} decoded grayscale {im.size}: {len(X)} full blocks")
            for tname, T in transforms.items():
                m = transform_metrics(X, T)
                print(
                    f"  {tname:10s} E16={m['E16']:.4f} E64={m['E64']:.4f} "
                    f"Huff={m['huffman_bits_per_coeff']:.4f} bits/coeff "
                    f"zero={m['zero_fraction']:.4f} RMSE={m['roundtrip_rmse_after_rounding']:.4f}"
                )
    except Exception as exc:
        print("Decoded image test skipped:", exc)
