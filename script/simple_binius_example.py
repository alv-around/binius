import numpy as np
import galois

from galois import GF
from utils import generate_extension, generate_merkletree, tensor_product

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
trace_ext = generate_extension(trace, F)
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
