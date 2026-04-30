"""
DEMO 4: Group Quantization with Per-Group Scales — Scientific Data Storage
═══════════════════════════════════════════════════════════════════════════

Used by: HDF5 with scale-offset filter, NetCDF/CF conventions (climate data),
         Zarr arrays (geospatial), seismic data formats (SEG-Y), Intel IMRS,
         Apache Arrow chunked arrays, time series databases (Gorilla, Prometheus).

TurboQuant's value quantization uses "group quantization":
  - Split the float32 value vector into groups of G elements
  - Compute scale and zero-point per group
  - Quantize each group to 2-bit (4 levels)
  - Store: indices (2-bit) + scale (float16) + zero (float16) per group

This is exactly how NetCDF stores climate model output (temperatures, pressures,
wind speeds) — too wide a dynamic range for uniform quantization to work well.

WHY groups?
  A 256-element vector may have values in [0.1, 0.2] in the first 32 elements
  and [10.0, 100.0] in the next 32. A single global scale wastes precision on
  the first group. Per-group scales give each chunk its own "zoom level."
"""

import numpy as np
import struct


# ─────────────────────────────────────────────
# GROUP QUANTIZATION
# ─────────────────────────────────────────────

def group_quantize(
    data: np.ndarray,
    group_size: int = 32,
    bits: int = 8
) -> tuple:
    """
    Quantize a float array in independent groups, each with its own scale.

    Returns:
        indices  — uint array of quantised indices (n_elements,)
        scales   — float32 array (n_groups,)
        zeros    — float32 array (n_groups,)  [minimum of each group]
    """
    n = len(data)
    n_levels = 2 ** bits
    n_groups = (n + group_size - 1) // group_size

    indices = np.zeros(n, dtype=np.uint16)
    scales = np.zeros(n_groups, dtype=np.float32)
    zeros = np.zeros(n_groups, dtype=np.float32)

    for g in range(n_groups):
        start = g * group_size
        end = min(start + group_size, n)
        chunk = data[start:end].astype(np.float64)

        mn, mx = chunk.min(), chunk.max()
        scale = (mx - mn) / (n_levels - 1) if mx > mn else 1.0
        zero = mn

        q = np.round((chunk - zero) / scale).clip(0, n_levels - 1).astype(np.uint16)
        indices[start:end] = q
        scales[g] = scale
        zeros[g] = zero

    return indices, scales, zeros


def group_dequantize(
    indices: np.ndarray,
    scales: np.ndarray,
    zeros: np.ndarray,
    group_size: int = 32
) -> np.ndarray:
    """Reconstruct float array from group-quantized representation."""
    n = len(indices)
    out = np.empty(n, dtype=np.float64)
    for g, (scale, zero) in enumerate(zip(scales, zeros)):
        start = g * group_size
        end = min(start + group_size, n)
        out[start:end] = indices[start:end].astype(np.float64) * scale + zero
    return out


# ─────────────────────────────────────────────
# STORAGE SIZING UTILITIES
# ─────────────────────────────────────────────

def storage_breakdown(n: int, group_size: int, bits: int) -> dict:
    """Calculate exact bytes used by group-quantized representation."""
    n_groups = (n + group_size - 1) // group_size
    # Bit-pack the indices: bits per element
    index_bytes = math.ceil(n * bits / 8)
    # Per-group metadata: scale (float16=2B) + zero (float16=2B)
    meta_bytes = n_groups * 4
    total = index_bytes + meta_bytes
    return {
        "index_bytes": index_bytes,
        "meta_bytes": meta_bytes,
        "total_bytes": total,
        "original_bytes": n * 4,  # float32
        "ratio": (n * 4) / total,
    }


import math

# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

def run_demo():
    print("=" * 60)
    print("DEMO 4: Group Quantization (Scientific / Climate Data)")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # Simulate a climate model variable: surface air temperature (Kelvin)
    # A realistic global 1-degree grid: 180 × 360 = 64,800 grid points
    # Temperature varies widely: poles ~220K, tropics ~300K, locally ±20K
    N = 64_800

    # Realistic temperature field: smooth global gradient + regional anomalies
    lats = np.linspace(-90, 90, 180)
    lons = np.linspace(0, 360, 360)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')

    base_temp = 288.0 - np.abs(lat_grid) * 0.8          # warmer at equator
    anomalies = (
        10 * np.sin(np.radians(lon_grid)) * np.cos(np.radians(lat_grid * 2)) +
        5 * rng.standard_normal((180, 360))               # weather noise
    )
    temperature = (base_temp + anomalies).flatten().astype(np.float32)

    print(f"\nVariable: Surface Air Temperature (float32), {N:,} grid points")
    print(f"  Range: [{temperature.min():.1f}, {temperature.max():.1f}] K")
    print(f"  Std:   {temperature.std():.2f} K")
    print(f"  Original size: {temperature.nbytes:,} bytes ({temperature.nbytes/1024:.1f} KB)")

    print("\n  Comparing group sizes at 8-bit quantization:")
    print(f"  {'Group size':>12}  {'Total bytes':>12}  {'Ratio':>6}  {'RMSE (K)':>10}  {'Max err (K)':>12}")
    print("  " + "─" * 60)

    for gs in [16, 32, 64, 128, 256]:
        q, scales, zeros = group_quantize(temperature, group_size=gs, bits=8)
        rec = group_dequantize(q, scales, zeros, group_size=gs).astype(np.float32)
        rmse = np.sqrt(np.mean((temperature - rec)**2))
        max_err = np.max(np.abs(temperature - rec))
        info = storage_breakdown(N, gs, 8)
        print(f"  {gs:>12}  {info['total_bytes']:>12,}  {info['ratio']:>5.2f}x  "
              f"{rmse:>10.4f}  {max_err:>12.4f}")

    # Detailed breakdown at group_size=32
    gs = 32
    q, scales, zeros = group_quantize(temperature, group_size=gs, bits=8)
    info = storage_breakdown(N, gs, 8)
    print(f"\n  Detailed breakdown (group_size=32, 8-bit):")
    print(f"    Index data:          {info['index_bytes']:>8,} bytes")
    print(f"    Scale+zero metadata: {info['meta_bytes']:>8,} bytes  ({info['meta_bytes']/info['total_bytes']*100:.1f}% overhead)")
    print(f"    Total compressed:    {info['total_bytes']:>8,} bytes")
    print(f"    Original float32:    {info['original_bytes']:>8,} bytes")
    print(f"    Compression ratio:   {info['ratio']:.2f}x")

    # ── Different bit widths ──
    print(f"\n  Compression vs quality at group_size=32:")
    print(f"  {'Bits':>5}  {'Bytes':>10}  {'Ratio':>6}  {'RMSE (K)':>10}  Notes")
    print("  " + "─" * 56)
    for bits in [2, 4, 8, 16]:
        q, scales, zeros = group_quantize(temperature, group_size=32, bits=bits)
        rec = group_dequantize(q, scales, zeros, group_size=32).astype(np.float32)
        rmse = np.sqrt(np.mean((temperature - rec)**2))
        info = storage_breakdown(N, 32, bits)
        note = {
            2:  "TurboQuant values",
            4:  "rough model output",
            8:  "NetCDF standard",
            16: "near-lossless"
        }[bits]
        print(f"  {bits:>5}  {info['total_bytes']:>10,}  {info['ratio']:>5.2f}x  {rmse:>10.4f}  {note}")

    # ── Contrast: global uniform quantization (no groups) fails here ──
    mn, mx = temperature.min(), temperature.max()
    for bits in [8]:
        scale_global = (mx - mn) / (2**bits - 1)
        q_global = np.round((temperature - mn) / scale_global).clip(0, 2**bits-1).astype(np.uint16)
        rec_global = q_global * scale_global + mn
        rmse_global = np.sqrt(np.mean((temperature.astype(np.float64) - rec_global)**2))
        q_group, scales_g, zeros_g = group_quantize(temperature, group_size=32, bits=bits)
        rec_group = group_dequantize(q_group, scales_g, zeros_g, group_size=32)
        rmse_group = np.sqrt(np.mean((temperature.astype(np.float64) - rec_group)**2))
        print(f"\n  Global uniform 8-bit RMSE:  {rmse_global:.4f} K")
        print(f"  Group quant  8-bit RMSE:    {rmse_group:.4f} K  ({rmse_global/rmse_group:.1f}x better)")

    print("\n  Connection to TurboQuant:")
    print("    • TQ splits the 128/256-dim value vector into groups of G=32")
    print("    • Each group gets its own float16 scale + zero point")
    print("    • Indices are 2-bit (4 levels) → bit-packed, 4 per byte")
    print("    • Storage: 2-bit/elem + 4 bytes/group metadata → ~198 bytes/token")
    print("    • Same tradeoff you see above: group_size 32 is the sweet spot")

    print("\n  Real-world usage:")
    print("    • NetCDF/CF: scale_factor + add_offset per variable (1 big group)")
    print("    • HDF5 scale-offset filter: per-chunk quantization")
    print("    • Zarr: per-shard quantization with codec pipelines")
    print("    • Prometheus: XOR delta compression on float64 gauge timeseries")
    print("    • GGUF (llama.cpp): Q4_K, Q8_K formats use per-32 group scales")


if __name__ == "__main__":
    run_demo()
