"""Tiny SVD arithmetic and inert worker forwarding; no numerical import or spawn."""

import io
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent / "analytics"))
from analytics import svd
from test_svd_summary import AnalyticTwoByTwo, Array


class Vector(list):
    def __mul__(self, other):
        if isinstance(other, Vector):
            return Vector(a * b for a, b in zip(self, other))
        return Vector(a * other for a in self)

    def __truediv__(self, scalar):
        return Vector(a / scalar for a in self)

    def tolist(self):
        return list(self)


class Matrix(Array):
    def reshape(self, rows, cols):
        return Matrix(super().reshape(rows, cols).rows)

    def ravel(self):
        return Vector(super().ravel())

    def __rmul__(self, scalar):
        return Matrix(super().__rmul__(scalar).rows)

    def __sub__(self, other):
        return Matrix(super().__sub__(other).rows)

    def __mul__(self, other):
        return Matrix(
            [[a * b for a, b in zip(x, y)] for x, y in zip(self.rows, other.rows)]
        )


class TinyBackend(AnalyticTwoByTwo):
    """Adapt the existing closed-form oracle to the older vector arithmetic API."""

    def asarray(self, values, dtype):
        assert dtype is self.float64
        return Matrix([list(values)])

    def outer(self, left, right):
        return Matrix(super().outer(left, right).rows)

    def sum(self, values):
        return sum(values.ravel() if isinstance(values, Matrix) else values)

    def svd(self, matrix, full_matrices):
        u, singular, vt = super().svd(matrix, full_matrices)
        return Matrix(u.rows), Vector(singular), Matrix(vt.rows)


class SvdNumericContracts(unittest.TestCase):
    def test_compute(self):
        expected = {
            "singular_values": [2.0, 1.0],
            "energy_fractions": [0.8, 0.2],
            "rank_one_residual_energy_fraction": 0.2,
            "rank_one_residual": [0.0, 0.0, 0.0, 1.0],
            "zero_energy": False,
        }
        backend = TinyBackend()
        values = [2.0, 0.0, 0.0, 1.0]
        result = svd.compute_with_numpy(backend, values, 2, 2, 7)
        self.assertEqual(backend.inputs, [values, values])
        self.assertEqual(
            json.dumps(result, allow_nan=False),
            json.dumps({"original": expected, "shuffled": expected}, allow_nan=False),
        )
        zero = {
            "singular_values": [0.0, 0.0],
            "energy_fractions": [None, None],
            "rank_one_residual_energy_fraction": None,
            "rank_one_residual": [0.0, 0.0, 0.0, 0.0],
            "zero_energy": True,
        }
        result = svd.compute_with_numpy(TinyBackend(), [0.0] * 4, 2, 2, 7)
        self.assertEqual(
            json.dumps(result), json.dumps({"original": zero, "shuffled": zero})
        )

    def test_worker(self):
        data = {"rows": 2, "cols": 1, "values": [2.0, 1.0], "seed": 7}
        backend = object()
        incoming = io.StringIO(json.dumps(data))
        with (
            mock.patch.object(svd, "_configure_worker") as configure,
            mock.patch.object(svd.sys, "stdin", incoming),
            mock.patch.object(incoming, "read", wraps=incoming.read) as read,
            mock.patch.object(svd.sys, "stdout", io.StringIO()) as output,
            mock.patch.dict(sys.modules, {"numpy": backend}),
            mock.patch.object(
                svd, "compute_with_numpy", return_value={"fixture": [1.0, None]}
            ) as compute,
        ):
            svd.worker()
        configure.assert_called_once_with()
        read.assert_called_once_with(200001)
        compute.assert_called_once_with(backend, [2.0, 1.0], 2, 1, 7)
        self.assertEqual(output.getvalue(), '{"fixture": [1.0, null]}\n')
        with (
            mock.patch.object(svd, "_configure_worker"),
            mock.patch.object(svd.sys, "stdin", io.StringIO(json.dumps(data))),
            mock.patch.object(svd.sys, "stdout", io.StringIO()),
            mock.patch.dict(sys.modules, {"numpy": backend}),
            mock.patch.object(
                svd, "compute_with_numpy", return_value={"fixture": float("nan")}
            ),
        ):
            with self.assertRaises(ValueError):
                svd.worker()


if __name__ == "__main__":
    unittest.main()
