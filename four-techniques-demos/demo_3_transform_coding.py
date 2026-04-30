"""
DEMO 3: Decorrelating Transforms Before Quantization — JPEG / DCT / Hadamard
══════════════════════════════════════════════════════════════════════════════

Used by: JPEG (DCT-II), H.264/H.265 video (integer DCT), JPEG2000 (Wavelet),
         802.11 Wi-Fi (OFDM is DCT-like), Hadamard codes in CDMA phones.

TurboQuant's rotation step is a random orthogonal transform. Its purpose:
"spread energy evenly across dimensions so scalar quantization works well."
This is *exactly* why JPEG applies DCT before quantizing image blocks.

Without a transform:
  Raw pixels are correlated (smooth gradients, edges repeat).
  A few "important" pixels carry most of the signal. Scalar quantization
  wastes bits on redundant, correlated information.

After DCT / orthogonal rotation:
  Energy is concentrated in a few independent coefficients.
  You can drop (quantize aggressively) the low-energy ones.
  → Same visual quality, fewer bytes.

Key math: any orthogonal matrix R preserves norms (||Rx|| = ||x||),
so after R^(-1) = R^T reconstruction, MSE is the same as before the transform.
The transform just makes that MSE distribute "better" across coordinates.
"""

import numpy as np
import math


# ─────────────────────────────────────────────
# DCT-II  (the transform inside JPEG)
# ─────────────────────────────────────────────

def dct_matrix(N: int) -> np.ndarray:
    """Build the N×N orthonormal DCT-II matrix."""
    k = np.arange(N)
    n = np.arange(N)
    D = np.cos(np.pi * (2*n[:, None] + 1) * k[None, :] / (2*N))
    D[:, 0] /= math.sqrt(N)
    D[:, 1:] /= math.sqrt(N / 2)
    return D.T  # shape (N, N) — rows are basis vectors


# ─────────────────────────────────────────────
# FAST WALSH-HADAMARD TRANSFORM
# O(N log N), no multiplications — used in CDMA, signal processing
# Also used as the rotation in some TurboQuant variants (Hadamard rotation)
# ─────────────────────────────────────────────

def fwht(x: np.ndarray) -> np.ndarray:
    """
    Fast Walsh-Hadamard Transform (in-place, normalised).
    N must be a power of 2.
    This is an orthogonal transform: H^T H = I (after scaling).
    """
    x = x.astype(np.float64).copy()
    N = len(x)
    assert (N & (N - 1)) == 0, "N must be power of 2"
    h = 1
    while h < N:
        for i in range(0, N, h * 2):
            for j in range(i, i + h):
                a, b = x[j], x[j + h]
                x[j], x[j + h] = a + b, a - b
        h *= 2
    return x / math.sqrt(N)


def ifwht(x: np.ndarray) -> np.ndarray:
    """Inverse FWHT — same as forward (Hadamard matrix is its own inverse after scaling)."""
    return fwht(x)


# ─────────────────────────────────────────────
# UNIFORM SCALAR QUANTIZATION (applied after transform)
# ─────────────────────────────────────────────

def quantize_scalar(x: np.ndarray, bits: int) -> tuple:
    """Uniform scalar quantization with per-array scale."""
    levels = 2 ** bits
    scale = (x.max() - x.min()) / (levels - 1) + 1e-12
    zero = x.min()
    q = np.round((x - zero) / scale).clip(0, levels - 1).astype(np.uint16)
    return q, scale, zero


def dequantize_scalar(q: np.ndarray, scale: float, zero: float) -> np.ndarray:
    return q.astype(np.float64) * scale + zero


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

def run_demo():
    print("=" * 60)
    print("DEMO 3: Decorrelating Transform Before Quantization (JPEG / Hadamard)")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # ── Scenario A: Correlated 1D signal (like adjacent pixels in a smooth gradient) ──
    N = 64
    t = np.linspace(0, 1, N)
    # Smooth signal: dominated by low frequencies — highly correlated
    signal = (
        0.8 * np.sin(2 * np.pi * 1.5 * t) +
        0.3 * np.sin(2 * np.pi * 5.0 * t) +
        0.05 * rng.standard_normal(N)
    )

    BITS = 4  # 4 bits per coefficient

    print(f"\nSignal: N={N} samples, {BITS}-bit quantization per value")
    print(f"Storage budget: {N * BITS} bits in all cases (same number of bits)")

    # ── Method 1: Quantize raw signal ──
    q_raw, s_raw, z_raw = quantize_scalar(signal, BITS)
    dec_raw = dequantize_scalar(q_raw, s_raw, z_raw)
    mse_raw = np.mean((signal - dec_raw) ** 2)
    snr_raw = 10 * np.log10(np.var(signal) / mse_raw)

    # ── Method 2: DCT → quantize → IDCT ──
    D = dct_matrix(N)
    coeffs_dct = D @ signal
    q_dct, s_dct, z_dct = quantize_scalar(coeffs_dct, BITS)
    dec_dct_coeffs = dequantize_scalar(q_dct, s_dct, z_dct)
    dec_dct = D.T @ dec_dct_coeffs
    mse_dct = np.mean((signal - dec_dct) ** 2)
    snr_dct = 10 * np.log10(np.var(signal) / mse_dct)

    # ── Method 3: Hadamard → quantize → inverse Hadamard ──
    pad = 1 << math.ceil(math.log2(N))
    sig_pad = np.zeros(pad)
    sig_pad[:N] = signal
    coeffs_had = fwht(sig_pad)[:N]
    q_had, s_had, z_had = quantize_scalar(coeffs_had, BITS)
    dec_had_coeffs = np.zeros(pad)
    dec_had_coeffs[:N] = dequantize_scalar(q_had, s_had, z_had)
    dec_had = ifwht(dec_had_coeffs)[:N]
    mse_had = np.mean((signal - dec_had) ** 2)
    snr_had = 10 * np.log10(np.var(signal) / mse_had)

    # ── Method 4: Random orthogonal rotation (TurboQuant's approach) ──
    # A random orthogonal matrix has the same decorrelating property
    A = rng.standard_normal((N, N))
    R, _ = np.linalg.qr(A)               # R is a random orthogonal matrix
    coeffs_rot = R @ signal
    q_rot, s_rot, z_rot = quantize_scalar(coeffs_rot, BITS)
    dec_rot_coeffs = dequantize_scalar(q_rot, s_rot, z_rot)
    dec_rot = R.T @ dec_rot_coeffs        # R^{-1} = R^T for orthogonal R
    mse_rot = np.mean((signal - dec_rot) ** 2)
    snr_rot = 10 * np.log10(np.var(signal) / mse_rot)

    print("\n  Method                   SNR (dB)   MSE")
    print("  " + "─" * 46)
    print(f"  Raw (no transform)       {snr_raw:6.1f}     {mse_raw:.6f}   ← baseline")
    print(f"  DCT-II (JPEG)            {snr_dct:6.1f}     {mse_dct:.6f}")
    print(f"  Hadamard (CDMA/fast)     {snr_had:6.1f}     {mse_had:.6f}")
    print(f"  Random orthogonal (TQ)   {snr_rot:6.1f}     {mse_rot:.6f}")

    # ── Why it works: energy compaction ──
    energy_raw = signal ** 2
    energy_dct = coeffs_dct ** 2
    top_k = 8  # top 8 of 64 coefficients
    raw_top8 = np.sort(energy_raw)[-top_k:].sum() / energy_raw.sum()
    dct_top8 = np.sort(energy_dct)[-top_k:].sum() / energy_dct.sum()
    print(f"\n  Energy compaction (what makes transforms work):")
    print(f"    Raw signal: top {top_k}/{N} values hold {raw_top8*100:.1f}% of energy")
    print(f"    DCT coeffs: top {top_k}/{N} coeffs hold {dct_top8*100:.1f}% of energy")
    print(f"    → DCT packs signal into fewer coefficients that need precision")
    print(f"    → Remaining coefficients can use coarser quantization")

    # ── Practical JPEG: zero out small DCT coefficients ──
    threshold = 0.1 * np.max(np.abs(coeffs_dct))
    sparse_dct = coeffs_dct.copy()
    sparse_dct[np.abs(sparse_dct) < threshold] = 0.0
    n_nonzero = np.count_nonzero(sparse_dct)
    dec_sparse = D.T @ sparse_dct
    mse_sparse = np.mean((signal - dec_sparse) ** 2)
    snr_sparse = 10 * np.log10(np.var(signal) / mse_sparse)
    print(f"\n  JPEG-style coefficient dropping (threshold={threshold:.3f}):")
    print(f"    Keep {n_nonzero}/{N} non-zero coefficients ({n_nonzero/N*100:.0f}% of data)")
    print(f"    SNR: {snr_sparse:.1f} dB  MSE: {mse_sparse:.6f}")
    print(f"    Effective compression: {N/n_nonzero:.1f}x with only {mse_sparse/mse_raw:.2f}x the error")

    print("\n  Connection to TurboQuant:")
    print("    • TQ's random orthogonal rotation is exactly Method 4")
    print("    • After rotation, each coordinate is ~i.i.d. Beta distributed")
    print("    • This makes per-coordinate Lloyd-Max quantization near-optimal")
    print("    • Reconstruction: R^T @ dequantize(indices) — same as IDCT")
    print("    • JPEG uses a fixed DCT; TQ uses a random R (no structured bias)")

    print("\n  Real-world usage:")
    print("    • JPEG: 8×8 DCT blocks, zigzag scan, quantize, Huffman code")
    print("    • H.264/H.265: 4×4 or 8×8 integer approximation of DCT")
    print("    • OFDM (Wi-Fi, LTE, 5G): IFFT is a type of orthogonal transform")
    print("    • CDMA (3G phones): Walsh-Hadamard spreading codes")
    print("    • PCA whitening: decorrelate sensor arrays before classification")


if __name__ == "__main__":
    run_demo()
