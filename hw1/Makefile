CXX = g++
CXXFLAGS = -std=c++17 -O3 -pthread -fopenmp
TBB_INCLUDE = -I/opt/ohpc/Taiwania3/pkg/intel/2024/oneapi/tbb/latest/include
TBB_LIB = -L/opt/ohpc/Taiwania3/pkg/intel/2024/oneapi/tbb/latest/lib/intel64/gcc4.8 -ltbb
TARGETS = hw1

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS)

hw1: hw1.cpp
	$(CXX) $(CXXFLAGS) $(TBB_INCLUDE) hw1.cpp $(TBB_LIB) -o hw1
