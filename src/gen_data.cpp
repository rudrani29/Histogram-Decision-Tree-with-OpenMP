// usage: ./gen_data N D out.bin
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>
int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: gen_data N D out.bin\n"); return 1; }
    int N = atoi(argv[1]), D = atoi(argv[2]);
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0, 1);
    std::uniform_real_distribution<float> ud(-1, 1);
    std::vector<float> X((size_t)N * D), y(N);
    for (int i = 0; i < N; ++i) {
        float* x = &X[(size_t)i * D];
        for (int f = 0; f < D; ++f) {
            if      (f < 10) x[f] = nd(rng);
            else if (f < 20) x[f] = ud(rng);
            else if (f == 20) x[f] = (float)(rng() % 10);
            else if (f >= 40) x[f] = std::exp(nd(rng));
            else x[f] = nd(rng);
        }
        y[i] = 3*x[0] + 2*x[1] - x[2] + 2*std::sin(3*x[3]) + 1.5f*x[4]*x[5] + 0.5f*nd(rng);
    }
    FILE* fp = fopen(argv[3], "wb");
    int hdr[2] = {N, D};
    fwrite(hdr, 4, 2, fp);
    fwrite(X.data(), 4, X.size(), fp);
    fwrite(y.data(), 4, y.size(), fp);
    fclose(fp);
}
