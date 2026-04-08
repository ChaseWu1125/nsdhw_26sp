#include <cstddef>
#include <algorithm>
#include <stdexcept> 
#include <pybind11/pybind11.h>
#include <pybind11/operators.h> 
#include <pybind11/stl.h> 
#include <mkl_cblas.h> 

namespace py = pybind11;

class Matrix {
private:
    size_t m_nrow;
    size_t m_ncol;
    double* m_buffer; 
public:
    // Constructor
    Matrix(size_t nrow, size_t ncol){
        m_nrow = nrow;
        m_ncol = ncol;
        m_buffer = new double[nrow * ncol];
        std::fill(m_buffer, m_buffer + nrow * ncol, 0.0);
    } 
    
    // Destructor
    ~Matrix() {
        delete[] m_buffer;
    }
    // Accessors
    size_t nrow() const { return m_nrow; }
    size_t ncol() const { return m_ncol; }
    double& operator()(size_t i, size_t j) { 
        if (i >= m_nrow || j >= m_ncol) {
            throw std::out_of_range("Matrix access out of range");
        }
        return m_buffer[i * m_ncol + j]; 
    }
    double operator()(size_t i, size_t j) const { 
        if (i >= m_nrow || j >= m_ncol) {
            throw std::out_of_range("Matrix access out of range");
        }
        return m_buffer[i * m_ncol + j]; 
    }
    double* get_buffer() const { return m_buffer; }

    // Compare operator
    bool operator==(const Matrix& other) const {
        if(m_nrow != other.m_nrow || m_ncol != other.m_ncol) return false;
        for(size_t i = 0; i< m_nrow * m_ncol; i++){
            // Using a small epsilon for floating point comparison
            if(std::abs(m_buffer[i] - other.m_buffer[i]) > 1e-9) return false;
        }
        return true;
    }

    // copy constructor
    Matrix(const Matrix& other) : m_nrow(other.m_nrow), m_ncol(other.m_ncol){
        m_buffer = new double[m_nrow * m_ncol];
        std::copy(other.m_buffer, other.m_buffer + m_nrow * m_ncol, m_buffer);
    }
    // copy assignment
    Matrix& operator=(const Matrix& other){
        if(this == &other) return *this;
        if(m_nrow != other.m_nrow || m_ncol != other.m_ncol){
            delete[] m_buffer;
            m_nrow = other.m_nrow;
            m_ncol = other.m_ncol;
            m_buffer = new double[m_nrow * m_ncol];
        }
        std::copy(other.m_buffer, other.m_buffer + m_nrow * m_ncol, m_buffer);
        return *this;
    }
    // move constructor
    Matrix(Matrix&& other) noexcept : m_nrow(other.m_nrow), m_ncol(other.m_ncol), m_buffer(other.m_buffer){
        other.m_nrow = 0;
        other.m_ncol = 0;
        other.m_buffer = nullptr;
    }
    // move assignment
    Matrix& operator=(Matrix&& other) noexcept{
        if(this == &other) return *this;
        delete[] m_buffer;
        m_nrow = other.m_nrow;
        m_ncol = other.m_ncol;
        m_buffer = other.m_buffer;
        other.m_nrow = 0;
        other.m_ncol = 0;
        other.m_buffer = nullptr;
        return *this;
    }
};

// Naive multiplication
Matrix multiply_naive(Matrix const &mat1, Matrix const &mat2){
    if(mat1.ncol() != mat2.nrow()){
        throw std::runtime_error("Incompatible matrix dimensions");
    }
    Matrix C(mat1.nrow(),mat2.ncol());
    for(size_t i=0; i<mat1.nrow(); i++){
        for(size_t j=0; j<mat2.ncol();j++){
            double sum = 0.0;
            for(size_t k=0; k<mat1.ncol();k++){
                sum += mat1(i,k) * mat2(k,j);
            }
            C(i,j) = sum;
        }
    }
    return C;
}

Matrix multiply_mkl(Matrix const &mat1, Matrix const &mat2){
    if(mat1.ncol() != mat2.nrow()){
        throw std::runtime_error("Incompatible matrix dimensions");
    }
    Matrix ret(mat1.nrow(), mat2.ncol());
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        mat1.nrow(), mat2.ncol(), mat1.ncol(), 1.0, 
        mat1.get_buffer(), mat1.ncol(), 
        mat2.get_buffer(), mat2.ncol(), 
        0.0, ret.get_buffer(), ret.ncol()
    );
    return ret;
}

// Tiled matrix multiplication
Matrix multiply_tile(const Matrix& A, const Matrix& B, size_t tile_size) {
    if (A.ncol() != B.nrow()) {
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication");
    }
    Matrix C(A.nrow(), B.ncol());

    for (size_t i0 = 0; i0 < A.nrow(); i0 += tile_size) {
        for (size_t k0 = 0; k0 < A.ncol(); k0 += tile_size) {
            for (size_t j0 = 0; j0 < B.ncol(); j0 += tile_size) {
                for (size_t i = i0; i < std::min(i0 + tile_size, A.nrow()); ++i) {
                    for (size_t k = k0; k < std::min(k0 + tile_size, A.ncol()); ++k) {
                        double r = A(i, k);
                        for (size_t j = j0; j < std::min(j0 + tile_size, B.ncol()); ++j) {
                            C(i, j) += r * B(k, j);
                        }
                    }
                }
            }
        }
    }
    return C;
}

PYBIND11_MODULE(_matrix, m){ 
    m.doc() = "Matrix class for homework 3";
    py::class_<Matrix>(m, "Matrix")
        .def(py::init<size_t, size_t>())
        .def_property_readonly("nrow", &Matrix::nrow)
        .def_property_readonly("ncol", &Matrix::ncol)
        .def("__getitem__", [](Matrix& m, py::tuple idx){
            if (idx.size() != 2) throw py::index_error("Index must be a 2-element tuple");
            return m(idx[0].cast<size_t>(), idx[1].cast<size_t>());
        })
        .def("__setitem__", [](Matrix& m, py::tuple idx, double val){
            if (idx.size() != 2) throw py::index_error("Index must be a 2-element tuple");
            m(idx[0].cast<size_t>(), idx[1].cast<size_t>()) = val;
        })
        .def(py::self == py::self); 
    m.def("multiply_naive", &multiply_naive);
    m.def("multiply_mkl", &multiply_mkl);
    m.def("multiply_tile", &multiply_tile);
}