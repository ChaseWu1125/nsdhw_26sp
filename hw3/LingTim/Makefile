# 設定編譯器
CXX = g++

# 取得 Python.h 的路徑
PYTHON_INCLUDES = $(shell python3-config --includes)
# 嘗試取得 pybind11 的路徑 (忽略找不到模組的錯誤)
PYBIND11_INCLUDES = $(shell python3 -m pybind11 --includes 2>/dev/null || echo "")

# --- 關鍵修改：明確指定 Linux 系統下的 MKL 路徑 ---
MKL_INCLUDES = -I/usr/include/mkl
# --------------------------------------------------

# 合併所有編譯參數
CXXFLAGS = -O3 -Wall -shared -std=c++11 -fPIC $(PYTHON_INCLUDES) $(PYBIND11_INCLUDES) $(MKL_INCLUDES)
PYTHON_SUFFIX = $(shell python3-config --extension-suffix)

# 目標檔名
TARGET = _matrix$(PYTHON_SUFFIX)

# --- 關鍵修改：明確指定 Linux 系統下的 MKL 函式庫路徑 ---
MKL_LIBS = -L/usr/lib/x86_64-linux-gnu -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
# --------------------------------------------------

.PHONY: all clean test

all: $(TARGET)

$(TARGET): matrix.cpp
	$(CXX) $(CXXFLAGS) matrix.cpp -o $(TARGET) $(MKL_LIBS)

test: $(TARGET)
	python3 -m pytest test_matrix.py -v

clean:
	rm -rf $(TARGET) __pycache__ .pytest_cache