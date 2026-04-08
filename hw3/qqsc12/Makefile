# 1. 自動偵測 Python 與 pybind11 的路徑
PYBIND11_INC := $(shell python3 -m pybind11 --include)
PYTHON_INC   := $(shell python3-config --includes)
PY_EXT_SUFFIX := $(shell python3-config --extension-suffix)

# 2. 編譯參數 (加入 CI 所需的路徑與你的硬體優化)
CXX = g++
CXXFLAGS = -O3 -march=native -Wall -shared -std=c++14 -fPIC \
           $(PYBIND11_INC) $(PYTHON_INC) \
           -I/usr/include/mkl

# 3. 連結參數
LDFLAGS = -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

# 4. 目標檔案
TARGET = _matrix$(PY_EXT_SUFFIX)

# 預設動作：編譯 TARGET
all: $(TARGET)

$(TARGET): _matrix.cpp
	$(CXX) $(CXXFLAGS) _matrix.cpp -o $(TARGET) $(LDFLAGS)

# 清理動作 (這就是剛才報錯找不到的地方)
clean:
	rm -rf $(TARGET) __pycache__ .pytest_cache performance.txt

# 測試動作
test: $(TARGET)
	PYTHONPATH=. python3 -m pytest test_matrix.py