#define CRV_IMPLEMENTATION
#define CRV_INTERNAL
#define CRV_TENSOR_CONV_TRANSPOSE1D_PRINT_ELAPSED_TIME
#include "../crave.h"

#define IDX3(d1, d2, D2, d3, D3) (d1 * D2 * D3) + (d2 * D3) + d3

void unnamed_1() {

}

void tensor_conv_transpose1d(tensor_t* x, tensor_t* w, size_t stride, size_t dilation) {
#ifdef CRV_TENSOR_CONV_TRANSPOSE1D_PRINT_ELAPSED_TIME
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
#endif

  CRV_DO_INTERNAL(
    crv_tensor_validate(x);
    crv_tensor_validate(w);
    assert(stride);
    assert(dilation);
    assert(x->rank == 3);
    assert(w->rank == 3);
  );

  //printf("x dims: [%zu, %zu, %zu], w dims: [%zu, %zu, %zu], stride: %zu, dilation: %zu\n",
  //  x->dims[0], x->dims[1], x->dims[2], w->dims[0], w->dims[1], w->dims[2], stride, dilation);

  size_t in_ch = w->dims[0];
  size_t out_ch = w->dims[1];
  size_t w_len = w->dims[2];

  size_t batches = x->dims[0];
  size_t x_in_ch = x->dims[1];
  size_t x_len = x->dims[2];
  assert(x_in_ch == in_ch);
  assert(!(w_len == 1 && dilation > 1));

  size_t eff_w_len = 1 + (w_len - 1) * dilation;
  size_t y_len = (x_len - 1) * stride + eff_w_len;

  float* x_data = x->data;
  float* w_data = w->data;
  float* y_data = x->swap;

  x->dims[1] = out_ch;
  x->dims[2] = y_len;
  x->count = x->dims[0] * x->dims[1] * x->dims[2];

  assert(x->count <= x->cap);

  memset(y_data, 0, x->count * sizeof(float));

  printf("batches: %zu, x_in_ch: %zu, x_len: %zu, out_ch: %zu, w_len: %zu\n", batches, x_in_ch, x_len, out_ch, w_len);

  for (size_t b = 0; b < batches; ++b) {
    for (size_t ic = 0; ic < x_in_ch; ++ic) {

      for (size_t i = 0; i < x_len; ++i) {

        for (size_t oc = 0; oc < out_ch; ++oc) {

          for (size_t k = 0; k < w_len; ++k) {

            size_t y_idx = (b * out_ch * y_len)  + (oc * y_len) + (i * stride) + (k * dilation);
            size_t x_idx = (b * x_in_ch * x_len) + (ic * x_len) + i;
            size_t w_idx = (ic * out_ch * w_len) + (oc * w_len) + k;
            y_data[y_idx] += x_data[x_idx] * w_data[w_idx];

            printf("Y: %zu, X: %zu, W: %zu\n", y_idx, x_idx, w_idx);

          }

        }

      }
    }
  }

  x->data = y_data;
  x->swap = x_data;

#ifdef CRV_TENSOR_CONV_TRANSPOSE1D_PRINT_ELAPSED_TIME
  clock_gettime(CLOCK_MONOTONIC, &end);
  double elapsed = CRV_ELAPSED_TIME(start, end);
  printf("tensor_conv_transposed1d() took: %.6fms\n", elapsed * 1e3);
#endif
}

size_t crv_file_get_size(FILE* file) {
  size_t result = 0;
  rewind(file);
  if (fseek(file, 0, SEEK_END) == 0) {
    result = ftell(file);
  }
  rewind(file);
  return result;
}

int main(void) {
  void* tensor_memory = malloc(200 * CRV_MB);

  if (tensor_memory) {

    char* it = (char*)tensor_memory;

    tensor_t* x1 = crv_tensor_create(&it, CRV_TPL(1, 192, 32), 16000, CRV_SWAP);
    tensor_t* w1 = crv_tensor_create(&it, CRV_TPL(192, 96, 8), CRV_TENSOR_AUTO_CAP, CRV_SWAP);
    tensor_t* x2 = crv_tensor_create(&it, CRV_TPL(1, 192, 32), 16000, CRV_SWAP);
    tensor_t* w2 = crv_tensor_create(&it, CRV_TPL(192, 96, 8), CRV_TENSOR_AUTO_CAP, CRV_SWAP);
    
    crv_tensor_randn(x2);
    crv_tensor_randn(w2);

    crv_tensor_copy(x1, x2);
    crv_tensor_copy(w1, w2);

    tensor_conv_transpose1d(x1, w1, 2, 1); 
    crv_tensor_conv_transpose1d(x2, w2, 2, 1);

    printf("\n");
    crv_tensor_print_error_stats(x1, x2);

  }

  free(tensor_memory);

  return 0;
}
