# Compiler settings
CXX = g++
CXXFLAGS = -O3 -march=native -fopenmp -std=c++17 -Iinclude
TARGET = hist_tree
SRC = $(wildcard src/*.cpp)

# Default target: builds the project
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Clean target: removes built binaries
clean:
	rm -f $(TARGET)