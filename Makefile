CXX=g++
CXXFLAGS=-O3 -march=native -fopenmp -std=c++17 -Wall -Isrc
A_SRC=src/split_final.cpp src/data.cpp src/tree_sort_version.cpp
B_SRC=src/hist.cpp src/tree_serial.cpp src/hist_omp.cpp
all: hist_tree gen_data test_a test_tree
hist_tree: src/main.cpp $(A_SRC) $(B_SRC) src/common.h src/tree.h
	$(CXX) $(CXXFLAGS) src/main.cpp $(A_SRC) $(B_SRC) -o hist_tree
gen_data: src/gen_data.cpp
	$(CXX) $(CXXFLAGS) src/gen_data.cpp -o gen_data
test_a: tests/test_a.cpp $(A_SRC) src/common.h
	$(CXX) $(CXXFLAGS) tests/test_a.cpp $(A_SRC) src/hist.cpp -o test_a
test_tree: tests/test_tree.cpp src/tree_serial.cpp src/hist.cpp src/split_final.cpp src/data.cpp
	$(CXX) $(CXXFLAGS) $^ -o test_tree
clean:
	rm -f hist_tree gen_data test_a test_tree
test_hist_omp: tests/test_hist_omp.cpp src/hist.cpp src/hist_omp.cpp src/common.h
	$(CXX) $(CXXFLAGS) tests/test_hist_omp.cpp src/hist.cpp src/hist_omp.cpp -o test_hist_omp