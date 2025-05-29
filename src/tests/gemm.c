#define CRV_IMPLEMENTATION
#include "../crave.h"
#include <omp.h>

#define MAX_N 2048
#define SIZE MAX_N * MAX_N

void crv_gepp(float* C, const float* A, const float* B, size_t M, size_t N, size_t K) {
  // NOTE(luca): C[M * N], A[M * K], B[K * N]

  for (size_t n = 0; n < N; ++n) {
    for (size_t k = 0; k < K; ++k) {
      for (size_t m = 0; m < M; ++m) {
        C[n * N + m] += A[n * N + k] * B[k * K + m];
      }
    }
  }
}

void crv_gemm(float* C, const float* A, const float* B, size_t M, size_t N, size_t K) {
  for () {
    for () {
      for () {

      }
    }
  }
}

void naive(float* C, const float* A, const float* B, size_t M, size_t N, size_t K) {
  // NOTE(luca): C[M * N], A[M * K], B[K * N]

  for (size_t n = 0; n < N; ++n) {
    for (size_t k = 0; k < K; ++k) {
      for (size_t m = 0; m < M; ++m) {
        C[n * N + m] += A[n * N + k] * B[k * K + m];
      }
    }
  }
}

int main() {
  srand(0);

  clock_t start;

  float* A = (float*)malloc(SIZE * sizeof(float));
  float* B = (float*)malloc(SIZE * sizeof(float));
  float* C1 = (float*)malloc(SIZE * sizeof(float));
  float* C2 = (float*)malloc(SIZE * sizeof(float));
  
  for (size_t i = 64; i <= MAX_N; i *= 2) {
    printf("Testing %zux%zu\n", i, i);

    for (size_t j = 0; j < SIZE; ++j) {
      A[j] = (float)rand() / (float)RAND_MAX;
      B[j] = (float)rand() / (float)RAND_MAX;
      C1[j] = 0;
      C2[j] = 0;
    }

    printf("Naive\n");
    start = clock();  
    naive(C1, A, B, i, i, i);
    crv_print_runtime_ms(start);

    printf("Opt\n");
    start = clock();  
    crv_gemm(C2, A, B, i, i, i);
    crv_print_runtime_ms(start);

    // VALIDATE
    float error = 0;
    for (size_t i = 0; i < SIZE; ++i) {
      error += fabs(C1[i] - C2[i]);
    }

    printf("L1 error: %f\n", error);
    printf("\n");
  }
  
  free(A);
  free(B);
  free(C1);
  free(C2);

  return 0;
}
