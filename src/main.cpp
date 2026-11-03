#include <iostream>
#include <omp.h>

int main() {
    std::cout << "OpenMP max threads: " << omp_get_max_threads() << std::endl;
    return 0;
}
