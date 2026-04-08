import pytest
import _matrix as mm 
import numpy as np 

# Helper to convert numpy array to our Matrix for test setup
def from_np(a):
    m = mm.Matrix(a.shape[0], a.shape[1])
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            m[i, j] = a[i, j]
    return m

# Test fixture for small matrices
@pytest.fixture
def small_mat_a():
    np_a = np.array([[1.0, 2.0], [3.0, 4.0]])
    return from_np(np_a)

@pytest.fixture
def small_mat_b():
    np_b = np.array([[5.0, 6.0], [7.0, 8.0]])
    return from_np(np_b)

@pytest.fixture
def small_mat_result_np():
    np_a = np.array([[1.0, 2.0], [3.0, 4.0]])
    np_b = np.array([[5.0, 6.0], [7.0, 8.0]])
    return np.dot(np_a, np_b) 

def test_matrix_creation_and_access():
    mat = mm.Matrix(2, 3)
    assert mat.nrow == 2
    assert mat.ncol == 3
    mat[0, 0] = 1.0
    mat[1, 2] = 5.0
    assert mat[0, 0] == 1.0
    assert mat[1, 2] == 5.0
    assert mat[0, 1] == 0.0 

def test_multiply_naive_correctness(small_mat_a, small_mat_b, small_mat_result_np):
    result_mm = mm.multiply_naive(small_mat_a, small_mat_b)
    
    # Convert mm.Matrix to numpy for comparison
    result_np_from_mm = np.zeros((result_mm.nrow, result_mm.ncol))
    for i in range(result_mm.nrow):
        for j in range(result_mm.ncol):
            result_np_from_mm[i, j] = result_mm[i, j]

    assert np.allclose(result_np_from_mm, small_mat_result_np)

def test_multiply_tile_correctness(small_mat_a, small_mat_b, small_mat_result_np):
    result_mm = mm.multiply_tile(small_mat_a, small_mat_b, 2) 
    
    result_np_from_mm = np.zeros((result_mm.nrow, result_mm.ncol))
    for i in range(result_mm.nrow):
        for j in range(result_mm.ncol):
            result_np_from_mm[i, j] = result_mm[i, j]

    assert np.allclose(result_np_from_mm, small_mat_result_np)

def test_multiply_mkl_correctness(small_mat_a, small_mat_b, small_mat_result_np):
    result_mm = mm.multiply_mkl(small_mat_a, small_mat_b)
    
    result_np_from_mm = np.zeros((result_mm.nrow, result_mm.ncol))
    for i in range(result_mm.nrow):
        for j in range(result_mm.ncol):
            result_np_from_mm[i, j] = result_mm[i, j]

    assert np.allclose(result_np_from_mm, small_mat_result_np)

def test_all_multiplications_match_each_other(small_mat_a, small_mat_b):
    naive_result = mm.multiply_naive(small_mat_a, small_mat_b)
    tile_result = mm.multiply_tile(small_mat_a, small_mat_b, 2)
    mkl_result = mm.multiply_mkl(small_mat_a, small_mat_b)

    assert naive_result == tile_result
    assert naive_result == mkl_result
    assert tile_result == mkl_result

def run_benchmark():
    """
    Run benchmark and write results to performance.txt.
    """
    size = 1024
    tile_size = 32

    print(f"Benchmarking with matrix size: {size}x{size}")
    print(f"Tiled implementation uses tile size: {tile_size}x{tile_size}")

    np_a = np.random.rand(size, size)
    np_b = np.random.rand(size, size)
    m_a = from_np(np_a)
    m_b = from_np(np_b)

    timings = {}

    # Time naive implementation
    start_time = time.time()
    mm.multiply_naive(m_a, m_b)
    end_time = time.time()
    timings['naive'] = end_time - start_time
    print(f"Naive multiplication took: {timings['naive']:.4f} seconds")

    # Time tiled implementation
    start_time = time.time()
    mm.multiply_tile(m_a, m_b, tile_size)
    end_time = time.time()
    timings['tile'] = end_time - start_time
    print(f"Tiled multiplication took: {timings['tile']:.4f} seconds")

    # Time MKL implementation
    start_time = time.time()
    mm.multiply_mkl(m_a, m_b)
    end_time = time.time()
    timings['mkl'] = end_time - start_time
    print(f"MKL multiplication took: {timings['mkl']:.4f} seconds")

    # Write performance.txt
    with open("performance.txt", "w") as f:
        f.write("Matrix Multiplication Performance\n")
        f.write("="*35 + "\n")
        f.write(f"Matrix size: {size}x{size}\n")
        f.write(f"Tile size: {tile_size}x{tile_size}\n\n")
        f.write(f"Naive implementation: {timings['naive']:.4f} seconds\n")
        f.write(f"Tiled implementation: {timings['tile']:.4f} seconds\n") 
        f.write(f"MKL implementation:   {timings['mkl']:.4f} seconds\n\n") 

        if timings['naive'] > 0:
            speedup = (timings['naive'] - timings['tile']) / timings['naive'] * 100
            f.write(f"Tiled version is {speedup:.2f}% faster than naive.\n")
            if speedup < 20:
                f.write("Warning: Tiled version is less than 20% faster than naive.\n")
            else:
                f.write("Success: Tiled version meets the 20% speedup requirement.\n")

if __name__ == "__main__":
    import time 
    run_benchmark()