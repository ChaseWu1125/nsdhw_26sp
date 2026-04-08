import os
import ctypes
import sys

# 針對 Linux 環境 (如 GitHub Actions) 預先載入 MKL 核心庫
if sys.platform.startswith('linux'):
    try:
        # 按照依賴順序載入，這通常能解決 undefined symbol 問題
        ctypes.CDLL("libmkl_rt.so", mode=ctypes.RTLD_GLOBAL)
    except Exception as e:
        print(f"MKL preload warning: {e}")

import _matrix
import pytest
import math

def test_basic():
    size = 100
    mat1 = _matrix.Matrix(size, size)
    mat2 = _matrix.Matrix(size, size)
    for i in range(size):
        for j in range(size):
            mat1[i, j] = i * size + j + 1
            mat2[i, j] = i * size + j + 1
    
    assert mat1.nrow == size
    assert mat1.ncol == size
    
    # 測試 Naive 與 MKL 結果是否一致
    ret_naive = _matrix.multiply_naive(mat1, mat2)
    ret_mkl = _matrix.multiply_mkl(mat1, mat2)
    
    for i in range(size):
        for j in range(size):
            assert math.isclose(ret_naive[i, j], ret_mkl[i, j], rel_tol=1e-9)

def test_tile():
    size = 200
    mat1 = _matrix.Matrix(size, size)
    mat2 = _matrix.Matrix(size, size)
    # 填充資料...
    ret_tile = _matrix.multiply_tile(mat1, mat2, 16)
    ret_mkl = _matrix.multiply_mkl(mat1, mat2)
    
    for i in range(size):
        for j in range(size):
            assert math.isclose(ret_tile[i, j], ret_mkl[i, j], rel_tol=1e-9)