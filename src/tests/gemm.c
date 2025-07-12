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

  printf("\n");
  test(768, 2, 2304);
  test(768, 2, 768);
  test(768, 2, 2304);
  test(768, 2, 768);
  test(384, 8, 1152);
  test(384, 8, 384);
  test(384, 8, 1152);
  test(384, 8, 384);
  test(384, 8, 1152);
  test(384, 8, 384);

  test(1, 1, 1);
  test(2, 3, 4);
  test(32, 32, 33);
  test(33, 32, 33);
  test(33, 34, 33);
  test(128, 128, 128);
  test(127, 128, 128);
  test(256, 256, 256);
  test(134, 1239, 2);
  test(512, 512, 512);
  test(1024, 1024, 1024);
  test(2048, 2048, 2048);
  test(2041, 2048, 2048);

  return 0;
}






























//
//#define IMPL_1
//
//#ifdef IMPL_0
//
//#define BLOCK 4
//#define BLOCK_M 1
//#define BLOCK_N 4
//
//void kernel_4x16(float* C, const float* A, const float* B, size_t n, size_t k) {
//  float32x4_t a[4];
//  float32x4_t b[4];
//  float32x4_t c[4][4] = {0};
//
//  for (size_t p = 0; p < k; ++p) {
//    b[0] = vld1q_f32(&B[p * n + 0]);
//    b[1] = vld1q_f32(&B[p * n + 4]);
//    b[2] = vld1q_f32(&B[p * n + 8]);
//    b[3] = vld1q_f32(&B[p * n + 12]);
//
//    a[0] = vdupq_n_f32(A[k * 0 + p]);
//    a[1] = vdupq_n_f32(A[k * 1 + p]);
//    a[2] = vdupq_n_f32(A[k * 2 + p]);
//    a[3] = vdupq_n_f32(A[k * 3 + p]);
//
//    c[0][0] = vfmaq_f32(c[0][0], a[0], b[0]);
//    c[0][1] = vfmaq_f32(c[0][1], a[0], b[1]);
//    c[0][2] = vfmaq_f32(c[0][2], a[0], b[2]);
//    c[0][3] = vfmaq_f32(c[0][3], a[0], b[3]);
//
//    c[1][0] = vfmaq_f32(c[1][0], a[1], b[0]);
//    c[1][1] = vfmaq_f32(c[1][1], a[1], b[1]);
//    c[1][2] = vfmaq_f32(c[1][2], a[1], b[2]);
//    c[1][3] = vfmaq_f32(c[1][3], a[1], b[3]);
//
//    c[2][0] = vfmaq_f32(c[2][0], a[2], b[0]);
//    c[2][1] = vfmaq_f32(c[2][1], a[2], b[1]);
//    c[2][2] = vfmaq_f32(c[2][2], a[2], b[2]);
//    c[2][3] = vfmaq_f32(c[2][3], a[2], b[3]);
//
//    c[3][0] = vfmaq_f32(c[3][0], a[3], b[0]);
//    c[3][1] = vfmaq_f32(c[3][1], a[3], b[1]);
//    c[3][2] = vfmaq_f32(c[3][2], a[3], b[2]);
//    c[3][3] = vfmaq_f32(c[3][3], a[3], b[3]);
//  }
//
//  vst1q_f32(&C[n * 0 + 0],  c[0][0]);
//  vst1q_f32(&C[n * 0 + 4],  c[0][1]);
//  vst1q_f32(&C[n * 0 + 8],  c[0][2]);
//  vst1q_f32(&C[n * 0 + 12], c[0][3]);
//
//  vst1q_f32(&C[n * 1 + 0],  c[1][0]);
//  vst1q_f32(&C[n * 1 + 4],  c[1][1]);
//  vst1q_f32(&C[n * 1 + 8],  c[1][2]);
//  vst1q_f32(&C[n * 1 + 12], c[1][3]);
//
//  vst1q_f32(&C[n * 2 + 0],  c[2][0]);
//  vst1q_f32(&C[n * 2 + 4],  c[2][1]);
//  vst1q_f32(&C[n * 2 + 8],  c[2][2]);
//  vst1q_f32(&C[n * 2 + 12], c[2][3]);
//
//  vst1q_f32(&C[n * 3 + 0],  c[3][0]);
//  vst1q_f32(&C[n * 3 + 4],  c[3][1]);
//  vst1q_f32(&C[n * 3 + 8],  c[3][2]);
//  vst1q_f32(&C[n * 3 + 12], c[3][3]);
//}
//
//void gemm(float* C, const float* A, const float* B, size_t m, size_t n, size_t k) {
////#pragma omp parallel for collapse(1) num_threads(10)
//
//  for (size_t i = 0; i < m; i += BLOCK_M * BLOCK) {
//    for (size_t j = 0; j < n; j += BLOCK_N * BLOCK) {
//      kernel_4x16(&C[i * n + j], &A[i * k], &B[j], n, k);
//    }
//  }
//  
//  //for (size_t i = 0; i < m; i += BLOCK_M * BLOCK) {
//  //  for (size_t j = 0; j < n; j += BLOCK_N * BLOCK) {
//  //    kernel_4x16(&C[i * n + j], &A[i * k], &B[j], n, k);
//  //  }
//  //}
//}
//#endif
//
//
//#ifdef IMPL_1
//
//#define TILE_M 64
//#define TILE_N 64
//#define TILE_K 32
//
//#define BLOCK 4
//#define BLOCK_M 1
//#define BLOCK_N 4
//
////void kernel(float* C, const float* A, const float* B, size_t m, size_t n, size_t k) {
////  for (size_t i = 0; i < m; i += BLOCK * BLOCK_M) {
////    for (size_t j = 0; j < n; j += BLOCK * BLOCK_N) {
////      
////
////      for (size_t p = 0; p < k; ++p) {
////
////        for (size_t ii = 0; ii < BLOCK * BLOCK_M; ++ii) {
////          float32x4_t a = vdupq_n_f32(A[(i + ii) * k + p]);
////
////          for (size_t jj = 0; jj < BLOCK_N * BLOCK; jj += BLOCK) {
////            float32x4_t b = vld1q_f32(&B[p * n + (j + jj)]);
////            float32x4_t c = vld1q_f32(&C[(i + ii) * n + (j + jj)]);
////
////            c = vfmaq_f32(c, a, b);
////
////            vst1q_f32(&C[(i + ii) * n + (j + jj)], c);
////          }
////        }
////
////      }
////    }
////  }
////}
//
