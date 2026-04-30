"""
DEMO 1: Bit-Packing in Database Column Stores
═══════════════════════════════════════════════

Used by: Apache Parquet, DuckDB, ClickHouse, Apache ORC, Roaring Bitmaps

The same idea TurboQuant uses to store 4 values per byte (2-bit packing)
is the backbone of every modern analytical database column store.

When you have a column of small integers (e.g. status codes, ratings 1-4,
enum types), storing each as a full 64-bit int is 32x wasteful.
Bit-packing collapses them to the minimum bits required.
"""

import sys
import time
import numpy as np


# ─────────────────────────────────────────────
# CORE: 2-bit packing (same as TurboQuant values)
# ─────────────────────────────────────────────

def pack_2bit(values: np.ndarray) -> bytes:
    """
    Pack an array of integers in [0, 3] into 2 bits each.
    4 values → 1 byte.  Used for 4-level enums, 2-bit quantized indices.
    """
    n = len(values)
    padded = np.zeros(((n + 3) // 4) * 4, dtype=np.uint8)
    padded[:n] = values
    # Interleave 4 values per byte using bitwise shifts
    packed = (
        (padded[0::4] & 0x3)        |   # bits 0-1
        ((padded[1::4] & 0x3) << 2) |   # bits 2-3
        ((padded[2::4] & 0x3) << 4) |   # bits 4-5
        ((padded[3::4] & 0x3) << 6)     # bits 6-7
    )
    return packed.tobytes()


def unpack_2bit(data: bytes, n: int) -> np.ndarray:
    """Recover the original integer array from 2-bit packed bytes."""
    packed = np.frombuffer(data, dtype=np.uint8)
    out = np.empty(len(packed) * 4, dtype=np.uint8)
    out[0::4] =  packed        & 0x3
    out[1::4] = (packed >> 2)  & 0x3
    out[2::4] = (packed >> 4)  & 0x3
    out[3::4] = (packed >> 6)  & 0x3
    return out[:n]


# ─────────────────────────────────────────────
# GENERALISED: arbitrary bit-width (1–8 bits)
# ─────────────────────────────────────────────

def pack_nbits(values: np.ndarray, bits: int) -> bytearray:
    """
    General N-bit packing via a bit-stream writer.
    bits=3 → 8 values per 3 bytes (used for 3-bit keys in TurboQuant)
    bits=4 → 2 values per byte  (used in Parquet for 16-level enums)
    """
    assert 1 <= bits <= 8
    buf = bytearray()
    pending, pending_bits = 0, 0
    mask = (1 << bits) - 1
    for v in values:
        pending |= (int(v) & mask) << pending_bits
        pending_bits += bits
        while pending_bits >= 8:
            buf.append(pending & 0xFF)
            pending >>= 8
            pending_bits -= 8
    if pending_bits > 0:
        buf.append(pending & 0xFF)
    return buf


def unpack_nbits(data: bytearray, n: int, bits: int) -> np.ndarray:
    """Reverse of pack_nbits."""
    mask = (1 << bits) - 1
    out = np.empty(n, dtype=np.int32)
    pending, pending_bits, byte_idx = 0, 0, 0
    for i in range(n):
        while pending_bits < bits and byte_idx < len(data):
            pending |= data[byte_idx] << pending_bits
            pending_bits += 8
            byte_idx += 1
        out[i] = pending & mask
        pending >>= bits
        pending_bits -= bits
    return out


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

def run_demo():
    print("=" * 60)
    print("DEMO 1: Bit-Packing  (Database Column Stores)")
    print("=" * 60)

    N = 1_000_000  # 1M rows — realistic for a DB column

    # Simulate an "order status" column: 0=pending 1=shipped 2=delivered 3=returned
    rng = np.random.default_rng(42)
    statuses = rng.integers(0, 4, size=N, dtype=np.uint8)

    # ── Baseline: uint8 (how most ORMs store it) ──
    baseline_bytes = statuses.nbytes
    print(f"\nColumn: {N:,} rows of order_status ∈ {{0,1,2,3}}")
    print(f"  Baseline  (uint8):  {baseline_bytes:>10,} bytes  ({baseline_bytes/1024:.1f} KB)")

    # ── 2-bit packing ──
    packed2 = pack_2bit(statuses)
    packed2_bytes = len(packed2)
    print(f"  2-bit packed:       {packed2_bytes:>10,} bytes  ({packed2_bytes/1024:.1f} KB)  "
          f"→ {baseline_bytes/packed2_bytes:.1f}x smaller")

    # Verify round-trip
    recovered = unpack_2bit(packed2, N)
    assert np.array_equal(statuses, recovered), "Round-trip failed!"
    print("  ✓ Round-trip verified (lossless)")

    # ── Generalised packing: 3-bit (0–7 priority levels) ──
    priorities = rng.integers(0, 8, size=N, dtype=np.uint8)
    packed3 = pack_nbits(priorities, bits=3)
    recovered3 = unpack_nbits(packed3, N, bits=3)
    assert np.array_equal(priorities, recovered3)
    print(f"\n  3-bit col (8 levels):  {priorities.nbytes:,} → {len(packed3):,} bytes  "
          f"→ {priorities.nbytes/len(packed3):.2f}x smaller")

    # ── Speed benchmark ──
    print("\n  Speed benchmark (pack + unpack 1M rows):")
    iters = 10
    t0 = time.perf_counter()
    for _ in range(iters):
        p = pack_2bit(statuses)
        _ = unpack_2bit(p, N)
    elapsed = (time.perf_counter() - t0) / iters * 1000
    print(f"    2-bit pack/unpack:  {elapsed:.2f} ms/call")

    # ── Real-world context ──
    print("\n  Real-world usage:")
    print("    • Apache Parquet uses bit-packing for RLE dictionary columns")
    print("    • DuckDB uses bit-packing natively in its column store format")
    print("    • ClickHouse bit-packs integer columns before disk I/O")
    print("    • Roaring Bitmaps pack 65536-element integer sets into 8KB")
    print("    • TurboQuant: 2-bit value packing, 3-bit key packing in KV cache")


if __name__ == "__main__":
    run_demo()
