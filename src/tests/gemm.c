#define CRV_IMPLEMENTATION
#include "../crave.h"

#include <stdalign.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>

#ifdef __APPLE__
 #define ACCELERATE_NEW_LAPACK
 #define ACCELERATE_LAPACK_ILP64
 #include <Accelerate/Accelerate.h>
 #define VALIDATE
#endif

void test(size_t m, size_t n, size_t k) {

  float* A = (float*)malloc(m * k * sizeof(float));
  float* B = (float*)malloc(k * n * sizeof(float));
  float* C1 = (float*)malloc(m * n * sizeof(float));
  float* C2 = (float*)malloc(m * n * sizeof(float));
   
  for (size_t i = 0; i < m * k; ++i) {
    A[i] = (float)rand() / (float)RAND_MAX;
  }

  for (size_t i = 0; i < k * n; ++i) {
    B[i] = (float)rand() / (float)RAND_MAX;
  }

  memset(C1, 0, m * n * sizeof(float));
  memset(C2, 0, m * n * sizeof(float));

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  crv_gemm(C1, n, A, k, B, n, m, n, k);
  clock_gettime(CLOCK_MONOTONIC, &end);
  
  {
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double gflops = (2.0 * m * n * k) / elapsed * 1e-9;

    char buf[32]; memset(buf, 0x20, 32);
    int len = sprintf(buf, "[m: %zu, n: %zu, k: %zu]", m, n, k); buf[len] = 0x20; buf[31] = 0;
    printf("%s %.1f GFLOPS\n", buf, gflops);
  }

#ifdef VALIDATE
  BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);

  clock_gettime(CLOCK_MONOTONIC, &start);
  cblas_sgemm(
    CblasRowMajor, CblasNoTrans, CblasNoTrans,
    m, n, k,
    1.0f,
    A, k,
    B, n,
    1.0f,
    C2, n
  );
  clock_gettime(CLOCK_MONOTONIC, &end);

  //{
  //  double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
  //  double gflops = (2.0 * m * n * k) / elapsed * 1e-9;

  //  char buf[32]; memset(buf, 0x20, 32);
  //  int len = sprintf(buf, "[m: %zu, n: %zu, k: %zu]", m, n, k); buf[len] = 0x20; buf[31] = 0;
  //  printf("%s %.1f GFLOPS\n", buf, gflops);
  //}

  float error = 0;
  for (size_t i = 0; i < m * n; ++i) {
    error += fabs(C1[i] - C2[i]);
  }

  if (error > 0.f) {
    fprintf(stderr, "[m: %zu, n: %zu, k: %zu] L1 error: %.12f\n", m, n, k, error);
  }
#endif

  free(A);
  free(B);
  free(C1);
  free(C2);
}

int main() {
  srand(0);

  test(128, 1, 128); 
  test(1536, 1, 384);
  //test(768, 2, 2304);
  //test(768, 2, 768); 
  //test(768, 2, 2304);
  //test(768, 2, 768); 
  //test(384, 8, 1152);
  //test(384, 8, 384); 
  //test(384, 8, 1152);
  //test(384, 8, 384); 
  //test(384, 8, 1152);
  //test(384, 8, 384); 
  //test(192, 32, 576);
  //test(192, 32, 192);
  //test(192, 32, 576);
  //test(192, 32, 192);
  //test(192, 32, 576);
  //test(192, 32, 192);
  //test(96, 128, 288);
  //test(96, 128, 96); 
  //test(96, 128, 288);
  //test(96, 128, 96); 
  //test(96, 128, 288);
  //test(96, 128, 96); 
  //test(128, 64, 384);
  //test(128, 32, 512);
  //test(80, 16, 512); 
  //test(32, 128, 672);
  //test(16, 128, 528);

  //printf("\n");
  //test(768, 2, 2304);
  //test(768, 2, 768);
  //test(768, 2, 2304);
  //test(768, 2, 768);
  //test(384, 8, 1152);
  //test(384, 8, 384);
  //test(384, 8, 1152);
  //test(384, 8, 384);
  //test(384, 8, 1152);
  //test(384, 8, 384);

  //test(1, 1, 1);
  //test(2, 3, 4);
  //test(32, 32, 33);
  //test(33, 32, 33);
  //test(33, 34, 33);
  //test(128, 128, 128);
  //test(127, 128, 128);
  //test(256, 256, 256);
  //test(134, 1239, 2);
  //test(512, 512, 512);
  //test(1024, 1024, 1024);
  //test(2048, 2048, 2048);
  //test(2041, 2048, 2048);

  return 0;
}
