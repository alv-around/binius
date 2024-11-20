from functools import reduce
import numpy as np
import galois
import hashlib
from galois import GF, Poly, Array
from merkly.mtree import MerkleTree


def sha_256(x: bytes, y: bytes) -> bytes:
    data = x + y
    h = hashlib.sha256()
    h.update(data)
    return h.digest()


def generate_extension(trace: Array, F: GF) -> Array:
    extension = []
    column_number = len(trace[0])
    xs = F.Range(0, column_number)
    for row in trace:
        f = galois.lagrange_poly(xs, row)
        ext = [f(x) for x in range(column_number, 2 * column_number)]
        extension.append(ext)

    return F(extension)


def generate_merkletree(trace: Array, ext: Array) -> MerkleTree:
    trace_depth = len(trace)
    assert trace_depth == len(ext) and len(trace[0]) == len(ext[0])
    assert (trace_depth & (trace_depth - 1) == 0) and trace_depth != 0

    circuit = np.append(trace, ext, axis=0)
    leafs = [str(Poly(columns)) for columns in circuit]
    return MerkleTree(leafs, hash_function=sha_256)


def tensor_product(values, F: GF):
    def cartesian_prod(x, y):
        return np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])

    tensor = reduce(lambda x, y: cartesian_prod(x, y), values)
    return F([np.prod(pairs) % F.order for pairs in tensor])
