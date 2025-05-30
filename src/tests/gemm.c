// NOTE(luca): clang src/tests/gemm.c -g -O3 -framework Accelerate -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 -o out/test_gemm && out/test_gemm
// NOTE(luca): C[m * n], A[m * k], B[k * n]

#define CRV_IMPLEMENTATION

#include "../crave.h"
#include <omp.h>
#include <Accelerate/Accelerate.h>

#define MAX_N 2048
#define TILE_SIZE 64 
#define SIZE MAX_N * MAX_N

void crv_gemm_t(float* C, const float* A, const float* B, size_t m, size_t n, size_t k) {
  for (size_t i = 0; i < m; ++i) {
    for (size_t p = 0; p < k; ++p) {
      for (size_t j = 0; j < n; ++j) {
        C[i * n + j] += A[i * k + p] * B[p * n + j];
      }
    }
  }
}

void crv_gemm(float* C, const float* A, const float* B, size_t m, size_t n, size_t k) {
  float A_[TILE_SIZE * TILE_SIZE];
  float B_[TILE_SIZE * TILE_SIZE];
  float C_[TILE_SIZE * TILE_SIZE] = {};

  for (size_t i = 0; i < m; i += TILE_SIZE) {
    for (size_t j = 0; j < n; j += TILE_SIZE) {
      memset(C_, 0, sizeof(float) * TILE_SIZE * TILE_SIZE);

      for (size_t p = 0; p < k; p += TILE_SIZE) {
        size_t A_idx = (i * k) + p;
        size_t B_idx = (p * n) + j;

        for (size_t _i = 0; _i < TILE_SIZE; ++_i) {
          memcpy(A_ + _i * TILE_SIZE, A + A_idx + _i * k, sizeof(float) * TILE_SIZE);
        }

        for (size_t _p = 0; _p < TILE_SIZE; ++_p) {
          memcpy(B_ + _p * TILE_SIZE, B + B_idx + _p * n, sizeof(float) * TILE_SIZE);
        }

        crv_gemm_t(C_, A_, B_, TILE_SIZE, TILE_SIZE, TILE_SIZE);
      }

      size_t C_idx = (i * n) + j;
      for (size_t _i = 0; _i < TILE_SIZE; ++_i) {
        for (size_t _j = 0; _j < TILE_SIZE; ++_j) {
          C[C_idx + _i * n + _j] += C_[_i * TILE_SIZE + _j];
        }
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
  
  for (size_t i = TILE_SIZE; i <= MAX_N; i *= 2) {
    printf("Testing %zux%zu\n", i, i);

    for (size_t j = 0; j < SIZE; ++j) {
      A[j] = (float)rand() / (float)RAND_MAX;
      B[j] = (float)rand() / (float)RAND_MAX;
      C1[j] = 0;
      C2[j] = 0;
    }

    printf("Mine:\n");
    start = clock();  
    crv_gemm(C1, A, B, i, i, i);
    crv_print_runtime_ms(start);

    printf("Accelerate:\n");
    start = clock();

    cblas_sgemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      i, i, i,
      1.0f,
      A, i,
      B, i,
      1.0f,
      C2, i 
    );

    crv_print_runtime_ms(start);

    // VALIDATE
    float error = 0;
    for (size_t j = 0; j < i * i; ++j) {
      error += fabs(C1[j] - C2[j]);
    }

    printf("l1 error: %f\n", error);
    printf("\n");
  }
  
  free(A);
  free(B);
  free(C1);
  free(C2);

  return 0;
}
