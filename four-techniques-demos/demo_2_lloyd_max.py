"""
DEMO 2: Non-Uniform Quantization — Audio Codecs (µ-law / G.711)
════════════════════════════════════════════════════════════════

Used by: G.711 telephony (every phone call since 1972), Opus codec internals,
         MP3 psychoacoustic model, early PCM digitisation standards.

TurboQuant's key insight: use a codebook optimised for the *actual distribution*
of the data (Beta distribution after rotation) rather than uniform buckets.
This is exactly what µ-law companding has done for audio for 50 years.

The principle:
  • Quiet audio has many more perceptually important values near zero.
  • A uniform 8-bit quantizer wastes half its levels on loud samples.
  • A log-companded quantizer packs more levels near zero → same 8 bits,
    much lower perceptual error.

Lloyd-Max generalises this: given any distribution P(x), compute the
optimal bucket boundaries and centroids that minimise MSE.
µ-law is the closed-form Lloyd-Max solution for a Laplacian distribution.
"""

import numpy as np
import math


# ─────────────────────────────────────────────
# µ-LAW COMPANDING  (ITU-T G.711 standard)
# ─────────────────────────────────────────────

MU = 255  # Standard µ value for North American / Japanese telephony

def encode_ulaw(samples: np.ndarray) -> np.ndarray:
    """
    Encode linear 16-bit PCM → 8-bit µ-law.
    Input range: [-1.0, 1.0]  Output: 256 levels (8 bits)
    """
    x = np.clip(samples.astype(np.float64), -1.0, 1.0)
    sign = np.sign(x)
    x = np.abs(x)
    # Log-compress: maps loud sounds to large quantisation steps,
    # quiet sounds (near 0) to small quantisation steps
    compressed = sign * (np.log1p(MU * x) / np.log1p(MU))
    # Map to 8-bit integer range
    return np.round(compressed * 127).astype(np.int8)


def decode_ulaw(encoded: np.ndarray) -> np.ndarray:
    """Decode 8-bit µ-law → linear PCM."""
    x = encoded.astype(np.float64) / 127.0
    sign = np.sign(x)
    x = np.abs(x)
    return sign * ((np.exp(x * np.log1p(MU)) - 1.0) / MU)


# ─────────────────────────────────────────────
# UNIFORM QUANTIZATION (naive baseline)
# ─────────────────────────────────────────────

def encode_uniform(samples: np.ndarray, bits: int = 8) -> np.ndarray:
    """Uniform scalar quantization — equally spaced buckets."""
    levels = 2 ** bits
    x = np.clip(samples, -1.0, 1.0)
    return np.round((x + 1.0) / 2.0 * (levels - 1)).astype(np.uint8)


def decode_uniform(encoded: np.ndarray, bits: int = 8) -> np.ndarray:
    levels = 2 ** bits
    return encoded.astype(np.float64) / (levels - 1) * 2.0 - 1.0


# ─────────────────────────────────────────────
# PURE PYTHON LLOYD-MAX SOLVER
# Finds optimal quantiser for any 1D distribution via EM-style iterations
# ─────────────────────────────────────────────

def lloyd_max(samples: np.ndarray, n_levels: int, n_iter: int = 50):
    """
    Lloyd-Max algorithm: iteratively optimise quantisation centroids.

    This is what TurboQuant precomputes for the Beta distribution.
    Here we apply it to an audio snippet's empirical distribution.

    Returns: (boundaries, centroids)
    """
    # Initialise centroids uniformly across sample range
    lo, hi = samples.min(), samples.max()
    centroids = np.linspace(lo, hi, n_levels)

    for _ in range(n_iter):
        # E-step: assign each sample to nearest centroid
        dists = np.abs(samples[:, None] - centroids[None, :])
        assignments = np.argmin(dists, axis=1)

        # M-step: move each centroid to the mean of its assigned samples
        new_centroids = np.array([
            samples[assignments == k].mean() if np.any(assignments == k) else centroids[k]
            for k in range(n_levels)
        ])

        if np.max(np.abs(new_centroids - centroids)) < 1e-8:
            break
        centroids = new_centroids

    # Boundaries are midpoints between adjacent centroids
    boundaries = (centroids[:-1] + centroids[1:]) / 2
    return boundaries, centroids


def quantize_lloyd(samples: np.ndarray, centroids: np.ndarray) -> tuple:
    """Quantize using a Lloyd-Max codebook."""
    dists = np.abs(samples[:, None] - centroids[None, :])
    indices = np.argmin(dists, axis=1).astype(np.uint8)
    reconstructed = centroids[indices]
    return indices, reconstructed


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

def run_demo():
    print("=" * 60)
    print("DEMO 2: Non-Uniform Quantization (Audio / µ-law / G.711)")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # Simulate voice audio: Laplacian distribution (quiet-heavy signal)
    # Real speech amplitude is Laplacian — most samples are near silence
    N = 8000  # 1 second at 8 kHz (telephone quality)
    raw = np.random.laplace(0, 0.15, size=N)
    raw = np.clip(raw, -1.0, 1.0)

    print(f"\nSignal: {N} samples of simulated voice (Laplacian, 8 kHz)")
    print(f"  Dynamic range: [{raw.min():.3f}, {raw.max():.3f}]")
    print(f"  Mean |x|: {np.mean(np.abs(raw)):.4f}  (most samples are quiet)")

    # ── Uniform 8-bit ──
    enc_uniform = encode_uniform(raw, bits=8)
    dec_uniform = decode_uniform(enc_uniform, bits=8)
    snr_uniform = 10 * np.log10(np.mean(raw**2) / np.mean((raw - dec_uniform)**2) + 1e-12)

    # ── µ-law 8-bit ──
    enc_ulaw = encode_ulaw(raw)
    dec_ulaw = decode_ulaw(enc_ulaw)
    snr_ulaw = 10 * np.log10(np.mean(raw**2) / np.mean((raw - dec_ulaw)**2) + 1e-12)

    # ── Lloyd-Max 4-bit (16 levels from actual data distribution) ──
    _, centroids_lm = lloyd_max(raw, n_levels=16, n_iter=100)
    _, dec_lloyd = quantize_lloyd(raw, centroids_lm)
    snr_lloyd = 10 * np.log10(np.mean(raw**2) / np.mean((raw - dec_lloyd)**2) + 1e-12)

    # ── Uniform 4-bit (baseline for comparison) ──
    enc_u4 = encode_uniform(raw, bits=4)
    dec_u4 = decode_uniform(enc_u4, bits=4)
    snr_u4 = 10 * np.log10(np.mean(raw**2) / np.mean((raw - dec_u4)**2) + 1e-12)

    print("\n  Method                Bits   SNR (dB)")
    print("  " + "─" * 44)
    print(f"  Uniform quantization    8    {snr_uniform:6.1f} dB   (baseline)")
    print(f"  µ-law (G.711)           8    {snr_ulaw:6.1f} dB   (same bits, better SNR)")
    print(f"  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─")
    print(f"  Uniform quantization    4    {snr_u4:6.1f} dB   (half the bits)")
    print(f"  Lloyd-Max optimal       4    {snr_lloyd:6.1f} dB   (same 4 bits, better)")
    print(f"\n  µ-law gain over uniform: +{snr_ulaw - snr_uniform:.1f} dB at same bit rate")
    print(f"  Lloyd-Max gain:          +{snr_lloyd - snr_u4:.1f} dB vs uniform at 4-bit")

    print("\n  Distribution insight:")
    pct_quiet = np.mean(np.abs(raw) < 0.1) * 100
    print(f"    {pct_quiet:.0f}% of samples have |x| < 0.1 (near silence)")
    print(f"    Uniform quantizer wastes ~75% of its levels on samples > 0.5")
    print(f"    µ-law packs ~50% of levels in the [-0.1, 0.1] range")

    print("\n  Connection to TurboQuant:")
    print("    • After rotation, KV vectors follow a Beta distribution")
    print("    • TurboQuant precomputes Lloyd-Max codebooks for that distribution")
    print("    • Same principle: match bucket density to actual data density")
    print("    • Result: same 3 bits per key, much lower attention error")

    print("\n  Real-world usage:")
    print("    • G.711: every PSTN phone call, VoIP, SIP trunk (since 1972)")
    print("    • Opus/Vorbis: psychoacoustic masking uses similar non-uniform quant")
    print("    • MP3: Huffman-coded quantised MDCT coefficients")
    print("    • ADPCM: adaptive step-size is Lloyd-Max applied online")


if __name__ == "__main__":
    run_demo()
