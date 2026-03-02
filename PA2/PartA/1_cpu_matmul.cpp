#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstdio>

void matmul_cpu(const float *a, const float *b, float *c, int n){
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            float sum = 0.0f;
            for(int k=0; k<n; k++){
                sum += a[i*n+k]*b[k*n+j];
            }
            c[i*n+j] = sum;
        }
    }
}

void init_matrix(float *m, int n){
    for(int i=0; i<n*n; i++){
        m[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main(int argc, char* argv[]){
    int n = 512;
    if(argc > 1){
        n = atoi(argv[1]);
    }
    size_t bytes = n*n*sizeof(float);
    float *a = (float*)malloc(bytes);
    float *b = (float*)malloc(bytes);
    float *c = (float*)malloc(bytes);
    init_matrix(a, n);
    init_matrix(b, n);
    printf("=====================================\n");
    printf("Starting cpu matmul for N=%d\n", n);
    const auto start_time = std::chrono::system_clock::now();
    matmul_cpu(a, b, c, n);
    const auto now = std::chrono::system_clock::now();
    printf("Time taken: %f s\n", (float)std::chrono::duration_cast<std::chrono::microseconds>(now-start_time).count()/1e6);
    printf("=====================================\n\n");
    free(a);
    free(b);
    free(c);
    return 0;
}
