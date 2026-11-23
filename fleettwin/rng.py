"""Counter-based random numbers that are bit-identical on CPU and GPU.

Each draw is a pure function of (seed, stream, counters...), so any
(scenario, step, cell or vehicle) can be generated independently, on any
device, with no stored state. The same scenario index gives the same draws
for every candidate plan (common random numbers), which makes plan
comparisons less noisy.

All arithmetic stays in int64 without overflow: 32-bit products are split
into 16-bit halves. The code only uses Python operators, so it runs
unchanged on Python ints and on torch tensors on any device.
"""

M32 = 0xFFFFFFFF
U32 = 1 << 32

# Streams
DEMAND = 1
DEST = 2
FAULT = 3
LEVEL = 4
EVENT = 5


def _mul32(x, c):
    """(x * c) mod 2**32 for 0 <= x, c < 2**32, without int64 overflow."""
    lo = (x & 0xFFFF) * c
    hi = ((x >> 16) * c) & 0xFFFF
    return (lo + (hi << 16)) & M32


def mix32(x):
    """MurmurHash3 32-bit finaliser (a bijection on 32-bit integers)."""
    x = x & M32
    x = x ^ (x >> 16)
    x = _mul32(x, 0x85EBCA6B)
    x = x ^ (x >> 13)
    x = _mul32(x, 0xC2B2AE35)
    x = x ^ (x >> 16)
    return x


def counter_hash(seed, stream, *counters):
    """Uniform integer in [0, 2**32) keyed by seed, stream and counters.

    Counters may be Python ints or int64 torch tensors;
    they broadcast against each other.
    """
    h = mix32((seed & M32) ^ _mul32(stream & M32, 0x9E3779B1))
    for c in counters:
        if isinstance(h, int):
            h = mix32(c ^ h)
        else:
            h = mix32(h ^ c)
    return h
