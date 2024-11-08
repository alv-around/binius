import numpy as np
import galois
import math
import hashlib
from galois import GF, Poly, Array
from merkly.mtree import MerkleTree
from typing import Callable


def sha_256(x: bytes, y: bytes) -> bytes:
    data = x + y
    h = hashlib.sha256()
    h.update(data)
    return h.digest()


def generate_extension(trace: Array) -> Array:
    extension = []
    column_number = len(trace[0])
    xs = F.Range(0, column_number)
    for row in trace:
        f = galois.lagrange_poly(xs, row)
        ext = [f(x) for x in range(column_number, 2 * column_number)]
        extension.append(ext)

    return F(extension)


def generate_merkle_root(trace: Array, ext: Array) -> MerkleTree:
    assert len(trace) == len(ext) and len(trace[0]) == len(ext[0])
    assert math.log2(len(trace)).is_integer()

    leafs = [str(Poly(columns)) for columns in trace + ext]
    return MerkleTree(leafs, hash_function=sha_256)


if __name__ == "__main__":
    F = GF(101)

    trace = F(
        [
            [3, 1, 4, 1],
            [5, 9, 2, 6],
            [5, 3, 5, 8],
            [9, 7, 9, 3],
        ]
    )

    trace_ext = generate_extension(trace)
    assert list(trace_ext[0]) == [
        F(82),
        F(34),
        F(48),
        F(12),
    ]  # [F(-19), F(-67), F(-154), F(-291)]

    root = generate_merkle_root(trace.T, trace_ext.T)
    print(root)
