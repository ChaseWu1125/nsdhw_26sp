# 1. 自動偵測 Python 路徑
PYTHON_INC   := $(shell python3-config --includes)
PY_EXT_SUFFIX := $(shell python3-config --extension-suffix)

# 2. 手動加上可能的 pybind11 路徑 (相容本地與 CI)
PYBIND11_INC := -I/usr/include -I/usr/local/include -I/usr/lib/python3/dist-packages/pybind11/include

# 3. 編譯參數
CXX = g++
CXXFLAGS = -O3 -march=native -Wall -shared -std=c++14 -fPIC \
           $(PYTHON_INC) $(PYBIND11_INC) \
           -I/usr/include/mkl

# 4. 連結參數
LDFLAGS = -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

# 5. 目標檔案
TARGET = _matrix$(PY_EXT_SUFFIX)

all: $(TARGET)

$(TARGET): _matrix.cpp
	$(CXX) $(CXXFLAGS) _matrix.cpp -o $(TARGET) $(LDFLAGS)

clean:
	rm -rf $(TARGET) __pycache__ .pytest_cache performance.txt

# 這裡加入 PYTHONPATH=. 確保 pytest 找得到 .so
test: $(TARGET)
	PYTHONPATH=. python3 -m pytest test_matrix.py