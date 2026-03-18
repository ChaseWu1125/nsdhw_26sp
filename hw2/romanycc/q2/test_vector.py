import _vector
import math
import pytest

def test_perpendicular():
    # (1, 0) 與 (0, 1) 應該是 90 度，即 pi/2
    angle = _vector.getAngle(1, 0, 0, 1)
    assert math.isclose(angle, math.pi / 2, rel_tol=1e-5)

def test_parallel():
    # (1, 1) 與 (2, 2) 方向相同，角度應為 0
    angle = _vector.getAngle(1, 1, 2, 2)
    assert math.isclose(angle, 0.0, abs_tol=1e-5)

def test_opposite():
    # (1, 0) 與 (-1, 0) 方向相反，角度應為 pi
    angle = _vector.getAngle(1, 0, -1, 0)
    assert math.isclose(angle, math.pi, rel_tol=1e-5)

def test_zero_vector():
    # 測試零向量是否會拋出錯誤 (如果你在 C++ 有寫 throw)
    with pytest.raises(RuntimeError):
        _vector.getAngle(0, 0, 1, 1)

def test_45_degree():
    assert math.isclose(_vector.getAngle(1, 0, 1, 1), math.pi / 4)