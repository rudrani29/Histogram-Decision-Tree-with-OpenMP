CXX=g++
CXXFLAGS=-O3 -march=native -fopenmp -std=c++17 -Wall
LIB=src/split.cpp src/data.cpp src/tree_exact.cpp src/tree_stub.cpp
all: hist_tree gen_data test_a
hist_tree: src/main.cpp $(LIB) src/common.h
	$(CXX) $(CXXFLAGS) src/main.cpp $(LIB) -o hist_tree
gen_data: src/gen_data.cpp
	$(CXX) $(CXXFLAGS) src/gen_data.cpp -o gen_data
test_a: src/test_a.cpp $(LIB) src/common.h
	$(CXX) $(CXXFLAGS) src/test_a.cpp $(LIB) -o test_a
clean:
	rm -f hist_tree gen_data test_a
