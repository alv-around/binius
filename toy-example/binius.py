from functools import reduce
import numpy as np
import galois
import hashlib
from galois import GF, Poly, Array
from merkly.mtree import MerkleTree
from sympy.physics.quantum import TensorProduct


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


def generate_merkletree(trace: Array, ext: Array) -> MerkleTree:
    trace_depth = len(trace)
    assert trace_depth == len(ext) and len(trace[0]) == len(ext[0])
    assert (trace_depth & (trace_depth - 1) == 0) and trace_depth != 0

    circuit = np.append(trace, ext, axis=0)
    leafs = [str(Poly(columns)) for columns in circuit]
    return MerkleTree(leafs, hash_function=sha_256)


def tensor_product(values, F):
    def cartesian_prod(x, y):
        return np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])

    tensor = reduce(lambda x, y: cartesian_prod(x, y), values)
    return F([np.prod(pairs) % F.order for pairs in tensor])


if __name__ == "__main__":
    # example values
    p = 101
    F = GF(p)
    rs = F.Range(1, 5)  # example random values chosen by verifier
    # example trace
    trace = F(
        [
            [3, 1, 4, 1],
            [5, 9, 2, 6],
            [5, 3, 5, 8],
            [9, 7, 9, 3],
        ]
    )
    n = len(trace[0])

    # pre-processing
    trace_ext = generate_extension(trace)
    trace_depth = len(trace)
    assert list(trace_ext[0]) == [
        F(82),
        F(34),
        F(48),
        F(12),
    ]  # [F(-19), F(-67), F(-154), F(-291)]
    tree = generate_merkletree(trace.T, trace_ext.T)
    print(tree)

    # proving
    ## create tensors
    rs_columns, rs_rows = np.split(rs, 2)
    columns = [[F(1) - fp, fp] for fp in rs_columns]
    rows = [[F(1) - fp, fp] for fp in rs_rows]
    tensor_col = tensor_product(columns, F)
    tensor_row = tensor_product(rows, F)
    assert list(tensor_row) == [
        (F(1) - F(3)) * (F(1) - F(4)),
        3 * (F(1) - F(4)),
        (F(1) - F(3)) * F(4),
        F(3) * F(4),
    ]

    t = np.dot(tensor_row, trace)
    assert list(t) == [F(41), F(86), F(74), F(25)]  # [F(41), F(-15), F(74), F(-76)]
    assert np.dot(t, tensor_col) == F(-137 % 101)
    poly_t = galois.lagrange_poly(F.Range(0, n), t)
    assert poly_t(7) == F(-10746 % p)
    assert poly_t(7) == np.dot(tensor_row, trace_ext[:, -1])
