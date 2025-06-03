#ifndef CRAVE_H
#define CRAVE_H

#include <stdio.h>
#include <time.h>
#include <stdint.h>

#define CRV_MAX_RANK 6
#define CRV_FRONT 0
#define CRV_BACK 1
#define CRV_MAX (1ULL << 48)
#define CRV_NO_SWAP 0
#define CRV_SWAP 1

#if __clang__
#define CRV_MAYBE_UNUSED __attribute__((unused))
#else
#define CRV_MAYBE_UNUSED [[maybe_unused]]
#endif
#define CRV_API CRV_MAYBE_UNUSED static

#ifdef __cplusplus
extern "C" {
#endif

// TODO(luca): Use size_t for some items such as count and cap.
typedef struct {
  uint32_t cap;
  uint32_t count;
  uint32_t rank;
  uint32_t dims[CRV_MAX_RANK];
  float* data;
  float* swap;
  char* name;
} tensor_t;

typedef struct {
  tensor_t** tensors;
  size_t count;
} tensor_list_t;

CRV_API inline void     crv_randn                             (float* output);

CRV_API inline void     crv_print_runtime_ms                  (clock_t start);
CRV_API inline void     crv_print_avg_runtime_ms              (clock_t start, uint32_t iters);

CRV_API inline void     crv_mm                                (float* dest, float* A, float* B, size_t cols, size_t rows, size_t len);

#ifdef CRV_INTERNAL
CRV_API inline void     crv_validate_tensor                   (tensor_t* input);
#endif

CRV_API inline void     crv_get_tensor_strides                (tensor_t* input, size_t* strides);
CRV_API inline size_t   crv_get_tensor_last_dim_index         (tensor_t* input);
CRV_API inline size_t   crv_get_tensor_last_dim_size          (tensor_t* input);

CRV_API size_t          crv_tensor_get_alloc_size_for_shape   (uint32_t* dims, uint32_t rank, int swap);
CRV_API tensor_t*       crv_tensor_create                     (char** dest, uint32_t* dims, uint32_t rank, uint32_t capacity, int swap);

CRV_API tensor_t*       crv_tensor_find_in_list               (tensor_list_t* list, const char* name);
CRV_API uint32_t        crv_read_u32_le                       (const char** it);
CRV_API void            crv_read_array                        (const char** it, void* data, size_t bytes);
CRV_API tensor_t*       crv_tensor_load_from_memory_iterator  (char** dest_it, const char** src_it);
CRV_API tensor_list_t*  crv_tensor_load_from_memory           (char** dest_it, const void* src, size_t count);

CRV_API void            crv_tensor_fill                       (tensor_t* tensor, float val);
CRV_API void            crv_tensor_hann                       (tensor_t* input);
CRV_API void            crv_tensor_mul                        (tensor_t* tensor, float mul);
CRV_API void            crv_tensor_add                        (tensor_t* tensor, float add);
CRV_API void            crv_tensor_tadd                       (tensor_t* dest, tensor_t* src);
CRV_API void            crv_tensor_pow                        (tensor_t* tensor, float pow);
CRV_API void            crv_tensor_arange                     (tensor_t* tensor, float start, float step);
CRV_API void            crv_tensor_cat                        (tensor_t* dest, tensor_t* src, uint32_t dim, int32_t direction);
CRV_API void            crv_tensor_pad                        (tensor_t* tensor, size_t left_pad, size_t right_pad);
CRV_API void            crv_tensor_trunc                      (tensor_t* tensor, uint32_t left_trunc, uint32_t right_trunc);
CRV_API void            crv_tensor_roll                       (tensor_t* input, int32_t shifts, size_t dim);
CRV_API void            crv_tensor_copy                       (tensor_t* dest, tensor_t* src);
CRV_API void            crv_tensor_squeeze                    (tensor_t* tensor, uint32_t dim);
CRV_API void            crv_tensor_unsqueeze                  (tensor_t* tensor, uint32_t dim);
CRV_API void            crv_tensor_transpose                  (tensor_t* tensor, uint32_t dim1, uint32_t dim2);
CRV_API void            crv_tensor_permute                    (tensor_t* tensor, uint32_t* dims, uint32_t rank);
CRV_API void            crv_tensor_flip                       (tensor_t* tensor, uint32_t dim);
CRV_API void            crv_tensor_snake                      (tensor_t* tensor, tensor_t* alpha);
CRV_API void            crv_tensor_leaky_relu                 (tensor_t* tensor, float alpha);
CRV_API void            crv_tensor_sigmoid                    (tensor_t* tensor);
CRV_API void            crv_tensor_tanh                       (tensor_t* tensor);
CRV_API void            crv_tensor_tmul                       (tensor_t* dest, tensor_t* src);
CRV_API void            crv_tensor_tmul_last_dim              (tensor_t* dest, tensor_t* src);
CRV_API void            crv_tensor_split                      (tensor_t* dest, tensor_t* src);
CRV_API void            crv_tensor_reshape                    (tensor_t* tensor, uint32_t* dims, uint32_t rank);
CRV_API void            crv_tensor_conv1d                     (tensor_t* x, tensor_t* w, uint32_t stride, uint32_t dilation);
CRV_API void            crv_tensor_conv_transpose1d           (tensor_t* x, tensor_t* w, uint32_t stride, uint32_t dilation);
CRV_API void            crv_tensor_rfft                       (tensor_t* input);
CRV_API void            crv_tensor_irfft                      (tensor_t* input);
CRV_API float           crv_tensor_l1_norm                    (tensor_t* a, tensor_t* b);
CRV_API float           crv_tensor_mae                        (tensor_t* a, tensor_t* b);
CRV_API float           crv_tensor_maxae                      (tensor_t* a, tensor_t* b);
CRV_API void            crv_tensor_print_error_stats          (tensor_t* a, tensor_t* b);
CRV_API void            crv_tensor_print_shape                (tensor_t* tensor);
CRV_API void            crv_tensor_print_data                 (tensor_t* tensor);
CRV_API void            crv_tensor_print                      (tensor_t* tensor);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CRAVE_H

// ################################################################################
// BEGIN IMPLEMENTATION

#ifdef CRV_IMPLEMENTATION

#include <assert.h> 
#include <stdlib.h>
#include <string.h>
#include <math.h>
//#include <Accelerate/Accelerate.h>

#ifdef CRV_INTERNAL
 #define CRV_DO_INTERNAL(func) do { func } while (0)
#else
 #define CRV_DO_INTERNAL(func)
#endif

#define CRV_KB 1000
#define CRV_MB 1000000
#define CRV_GB 1000000000
#define CRV_TENSOR_AUTO_CAP 0

#define CRV_TPL(...) \
  (uint32_t[]) {__VA_ARGS__}, sizeof((uint32_t[]) {__VA_ARGS__}) / sizeof(uint32_t)

void crv_print_runtime_ms(clock_t start) {
  clock_t end = clock();
  double runtime = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
  printf("Runtime: %.4fms\n", runtime);
}

void crv_print_avg_runtime_ms(clock_t start, uint32_t iters) {
  clock_t end = clock();
  double runtime = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
  printf("Average runtime: %.4fms\n", runtime / (double)iters);
}

void crv_randn(float* output) {
  float u1 = ((float)rand() + 1.f) / ((float)RAND_MAX + 1.f);
  float u2 = ((float)rand() + 1.f) / ((float)RAND_MAX + 1.f);
  float radius = sqrtf(-2.f * logf(u1));
  float theta = 2.f * M_PI * u2;

  output[0] = radius * cosf(theta);
  output[1] = radius * sinf(theta);
}

void crv_mm(float* dest, float* A, float* B, size_t cols, size_t rows, size_t len) {
  for (size_t k = 0; k < rows; ++k) {
    for (size_t i = 0; i < cols; ++i) {
      for (size_t j = 0; j < len; ++j) {
        dest[i * len + j] += A[k * len + j] * B[i * rows + k];
      }
    }
  }
}

#ifdef CRV_INTERNAL
void crv_validate_tensor(tensor_t* input) {
  assert(input != NULL);
  assert(input->data != NULL);
  assert(input->rank > 0);
  assert(input->rank <= CRV_MAX_RANK);
  assert(input->count > 0);
  assert(input->count <= input->cap);
  assert(input->cap > 0);

  size_t count = 1;
  for (size_t i = 0; i < input->rank; ++i) {
    count *= input->dims[i];
  }

  assert(count == input->count);
}
#endif

void crv_get_tensor_strides(tensor_t* tensor, size_t* strides) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(tensor);
    assert(strides != NULL);
  );
  size_t rank = tensor->rank;
  uint32_t* dims = tensor->dims;
  strides[rank - 1] = 1;
  for (size_t i = rank - 1; i > 0; --i) {
    strides[i - 1] = strides[i] * dims[i]; 
  }
}

size_t crv_get_tensor_last_dim_index(tensor_t* input) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(input);
  );

  return input->rank - 1;
}

size_t crv_get_tensor_last_dim_size(tensor_t* input) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(input);
  );

  return input->dims[crv_get_tensor_last_dim_index(input)];
}

size_t crv_tensor_get_alloc_size_for_shape(uint32_t* dims, uint32_t rank, int swap) {
  size_t size = 1;
  for (size_t i = 0; i < rank; ++i) {
    size *= dims[i];
  }

  if (swap == CRV_SWAP) {
    size *= 2;
  }

  return size * sizeof(float) + sizeof(tensor_t);
}

tensor_t* crv_tensor_create(char** dest, uint32_t* dims, uint32_t rank, uint32_t capacity, int swap) {
  assert(dest);
  assert(rank > 0);
  assert(dims != NULL);

  tensor_t* tensor = (tensor_t*)*dest;
  *dest += sizeof(tensor_t);

  tensor->rank = rank;
  tensor->count = 1;

  for (uint32_t i = 0; i < rank; ++i) {
    assert(dims[i]);
    tensor->dims[i] = dims[i];
    tensor->count *= dims[i];
  }

  if (capacity == CRV_TENSOR_AUTO_CAP) {
    tensor->cap = tensor->count;
  } else {
    tensor->cap = capacity;
    assert(tensor->count <= capacity);
  }

  tensor->data = (float*)*dest;
  *dest += tensor->cap * sizeof(float);

  if (swap == CRV_SWAP) {
    tensor->swap = (float*)*dest;
    *dest += tensor->cap * sizeof(float);
  }

  return tensor;
}

void crv_tensor_init(tensor_t* tensor, uint32_t* dims, uint32_t rank) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(tensor);
    assert(dims != NULL);
    assert(rank > 0);
  );

  size_t count = 1;
  for (size_t i = 0; i < rank; ++i) {
    tensor->dims[i] = dims[i];
    count *= dims[i];
  }
  
  assert(count <= tensor->cap);

  tensor->count = count;
  tensor->rank = rank;
}

tensor_t* crv_tensor_find_in_list(tensor_list_t* list, const char* name) {
  for (size_t i = 0; i < list->count; ++i) {
    if (strcmp(list->tensors[i]->name, name) == 0) {
      return list->tensors[i];
    }
  }

  return NULL;
}

uint32_t crv_read_u32_le(const char** it) {
  const uint8_t* data = (const uint8_t*)*it;
  uint32_t value = (data[0] << 0) | (data[1] << 8) | (data[2] << 16) | (data[3] << 24);
  *it += sizeof(value);
  return value;
}

void crv_read_array(const char** it, void* data, size_t bytes) {
  memcpy(data, *it, bytes);
  *it += bytes;
}

tensor_t* crv_tensor_load_from_memory_iterator(char** dest_it, const char** src_it) {
  uint32_t name_len = crv_read_u32_le(src_it);

  char* name = (char*)*dest_it;
  *dest_it += name_len * sizeof(char);
  crv_read_array(src_it, name, name_len * sizeof(*name));

  uint32_t rank = crv_read_u32_le(src_it);
  uint32_t dims[CRV_MAX_RANK];
  crv_read_array(src_it, dims, rank * sizeof(dims[0]));

  uint32_t item_count = crv_read_u32_le(src_it);

  tensor_t* tensor = crv_tensor_create(dest_it, dims, rank, CRV_TENSOR_AUTO_CAP, CRV_NO_SWAP);
  crv_read_array(src_it, tensor->data, item_count * sizeof(*tensor->data));

  tensor->name = name;

  return tensor;
}

tensor_list_t* crv_tensor_load_from_memory(char** dest_it, const void* src, size_t count) {
  const char* src_it = (const char*)src;

  tensor_list_t* list = (tensor_list_t*)*dest_it;
  *dest_it += sizeof(tensor_list_t*);

  list->tensors = (tensor_t**)*dest_it;
  *dest_it += count * sizeof(tensor_t*);
  list->count = count;

  for (int i = 0; i < count; ++i) {
    list->tensors[i] = crv_tensor_load_from_memory_iterator(dest_it, &src_it);
  }

  return list;
}

void crv_tensor_fill(tensor_t* tensor, float val) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(tensor);
  );

  for (size_t i = 0; i < tensor->count; ++i) {
    tensor->data[i] = val;
  }
}

void crv_tensor_hann(tensor_t* input) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(input);
    assert(input->rank == 1);
    assert(input->dims[0] > 1);
  );

  size_t N = input->dims[0];
  for (size_t n = 0; n < N; ++n) {
    input->data[n] = 0.5f * (1.f - cosf((2.f * (float)M_PI * (float)n) / (N - 1)));
  }
}

void crv_tensor_mul(tensor_t* tensor, float mul) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(tensor);
  );

  for (uint32_t i = 0; i < tensor->count; ++i) {
    tensor->data[i] *= mul;
  }
}

void crv_tensor_pow(tensor_t* tensor, float pow) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(tensor);
  );

  for (uint32_t i = 0; i < tensor->count; ++i) {
    tensor->data[i] = powf(tensor->data[i], pow);
  }
}

void crv_tensor_add(tensor_t* tensor, float add) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(tensor);
  );

  for (uint32_t i = 0; i < tensor->count; ++i) {
    tensor->data[i] += add;
  }
}

void crv_tensor_tadd(tensor_t* dest, tensor_t* src) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(src);
    crv_validate_tensor(dest);
    assert(src->rank == dest->rank && "Tensor rank must match.");
    assert(src->count == dest->count && "Tensor item count must match.");
  );

  for (size_t i = 0; i < dest->count; ++i) {
    dest->data[i] += src->data[i];
  }
}

void crv_tensor_arange(tensor_t* tensor, float start, float step) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(tensor);
  );

  for (size_t i = 0; i < tensor->count; ++i) {
    tensor->data[i] = (float)i * step + start;
  }
}

void crv_tensor_cat(tensor_t* dest, tensor_t* src, uint32_t dim, int32_t direction) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(src);
    crv_validate_tensor(dest);
    assert(src->rank == dest->rank);
    assert(dim < dest->rank);
    assert(direction == CRV_FRONT || direction == CRV_BACK);

    for (uint32_t i = 0; i < dest->rank; ++i) {
      if (i != dim) {
        assert(dest->dims[i] == src->dims[i]);
      }
    }
  );

  size_t rank = dest->rank;
  size_t a_cpy_size, b_cpy_size;
  float* out = dest->swap;
  float* a;
  float* b;

  if (direction == CRV_BACK) {
    a_cpy_size = dest->dims[dim];
    b_cpy_size = src->dims[dim];

    for (uint32_t i = dim + 1; i < rank; ++i) {
      a_cpy_size *= dest->dims[i];
      b_cpy_size *= src->dims[i];
    }

    a = dest->data;
    b = src->data;
  } else {
    a_cpy_size = src->dims[dim];
    b_cpy_size = dest->dims[dim];

    for (uint32_t i = dim + 1; i < rank; ++i) {
      a_cpy_size *= src->dims[i];
      b_cpy_size *= dest->dims[i];
    }

    a = src->data;
    b = dest->data;
  }

  size_t copied_count = 0;
  size_t total_items = dest->count + src->count;
  assert(total_items <= dest->cap);

  size_t i = 0;
  while (copied_count < total_items) {
    memcpy(&out[copied_count], &a[i * a_cpy_size], sizeof(float) * a_cpy_size);
    copied_count += a_cpy_size;
    memcpy(&out[copied_count], &b[i * b_cpy_size], sizeof(float) * b_cpy_size);
    copied_count += b_cpy_size;
    ++i;
  }

  assert(copied_count == total_items);
  dest->dims[dim] += src->dims[dim];

  dest->count = total_items;
  dest->swap = dest->data;
  dest->data = out;
}

void crv_tensor_pad(tensor_t* tensor, size_t left_pad, size_t right_pad) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(tensor);
    assert(left_pad > 0 || right_pad > 0);
  );

  float* x = tensor->data;
  float* y = tensor->swap;

  size_t rank = tensor->rank;
  size_t len = tensor->dims[tensor->rank - 1];

  size_t total_dims = 1;
  for (uint32_t i = 0; i < rank - 1; ++i) {
    total_dims *= tensor->dims[i];
  }

  assert(total_dims * (left_pad + right_pad + len) <= tensor->cap);

  for (size_t i = 0; i < total_dims; ++i) {
    for (size_t j = 0; j < left_pad; ++j) {
      y[i * (len + left_pad + right_pad) + j] = 0.f;
    }

    for (size_t j = 0; j < right_pad; ++j) {
      y[i * (len + left_pad + right_pad) + left_pad + len + j] = 0.f;
    }

    memcpy(
      &y[i * (len + left_pad + right_pad) + left_pad],
      &x[i * len],
      len * sizeof(float)
    );
  }

  tensor->data = y;
  tensor->swap = x;
  tensor->count = total_dims * (len + left_pad + right_pad);
  tensor->dims[rank - 1] += left_pad + right_pad;
}

void crv_tensor_trunc(tensor_t* tensor, uint32_t left_trunc, uint32_t right_trunc) {
  // TODO(luca): Only implements left and right truncation.

  CRV_DO_INTERNAL(
    crv_validate_tensor(tensor);
    assert(!(left_trunc == 0 && right_trunc == 0) &&
      "No point in truncating if both values are zero.");
  );
  
  size_t count = tensor->count;
  size_t rank = tensor->rank;
  size_t x_len = tensor->dims[rank - 1];
  assert(x_len > left_trunc + right_trunc && "Tensor too small for desired truncation.");
  assert(count % x_len == 0 &&
    "If the last dimension isn't a multiple of the item_count then there is something wrong.");

  size_t y_len = x_len - left_trunc - right_trunc;   

  float* x = tensor->data;
  float* y = tensor->swap;

  for (size_t r = 0, w = 0; r < count;) {
    memcpy(&y[w], &x[r + left_trunc], y_len * sizeof(float)); 
    r += x_len;
    w += y_len;
  }

  tensor->dims[rank - 1] = y_len;  
  tensor->data = y;
  tensor->swap = x;
  tensor->count = count / x_len * y_len;
}

void crv_tensor_roll(tensor_t* input, int32_t shift, size_t dim) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(input);
    assert(shift != 0);
    assert(shift < (int32_t)input->dims[dim]);
    assert(dim > 0);
  );

  size_t strides[CRV_MAX_RANK]; 
  crv_get_tensor_strides(input, &strides[0]);

  float* x = input->data;
  float* y = input->swap;
  size_t count = input->count;
  size_t rank = input->rank;

  size_t indices[CRV_MAX_RANK];
  for (size_t i = 0; i < count; ++i) {
    size_t tmp = i;
    for (size_t j = 0; j < rank; ++j) {
      indices[j] = tmp / strides[j];
      tmp %= strides[j];
    }

    if (indices[dim] + shift < 0) {
      indices[dim] += shift + input->dims[dim];
    } else if (indices[dim] + shift > input->dims[dim]) {
      indices[dim] += shift - input->dims[dim]; 
    } else {
      indices[dim] += shift; 
    }

    indices[dim] = ((long)indices[dim] + shift) % (long)input->dims[dim];

    size_t write_index = 0;
    for (size_t j = 0; j < rank; ++j) {
      write_index += indices[j] * strides[j];
    }

    y[write_index] = x[i];
  }

  input->data = y;
  input->swap = x;
}

void crv_tensor_copy(tensor_t* dest, tensor_t* src) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(src);
    crv_validate_tensor(dest);
    assert(src->count <= dest->cap);
  );

  size_t item_count = src->count;

  float* x = src->data;
  float* y = dest->data;
  memcpy(y, x, item_count * sizeof(float));

  assert(src->rank > 0);
  assert(src->dims != NULL);
  assert(dest->dims != NULL);
  memcpy(dest->dims, src->dims, src->rank * sizeof(uint32_t));

  dest->count = item_count;
  dest->rank = src->rank;
  dest->name = src->name;
}

void crv_tensor_squeeze(tensor_t* tensor, uint32_t dim) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(tensor);
    assert(dim <= tensor->rank);
    assert(tensor->dims[dim] == 1);
  );

  size_t rank = tensor->rank;
  for (size_t i = dim; i < rank - 1; --i) {
    tensor->dims[i] = tensor->dims[i + 1];
  }

  tensor->rank = rank - 1;
}

void crv_tensor_unsqueeze(tensor_t* tensor, uint32_t dim) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(tensor);
    assert(dim <= tensor->rank);
  );

  size_t rank = tensor->rank;
  for (size_t i = rank; i > dim; --i) {
    tensor->dims[i] = tensor->dims[i - 1];
  }

  tensor->dims[dim] = 1;
  tensor->rank = rank + 1;
}

void crv_tensor_transpose(tensor_t* tensor, uint32_t dim1, uint32_t dim2) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(tensor);
    assert(dim1 < tensor->rank);
    assert(dim2 < tensor->rank);
  );

  size_t rank = tensor->rank;
  size_t item_count = tensor->count;

  size_t old_dims[CRV_MAX_RANK] = {};
  size_t new_dims[CRV_MAX_RANK] = {};

  for (size_t i = 0; i < rank; ++i) {
    old_dims[i] = tensor->dims[i];
    new_dims[i] = tensor->dims[i];
  }

  new_dims[dim1] = old_dims[dim2];
  new_dims[dim2] = old_dims[dim1];

  size_t old_strides[CRV_MAX_RANK] = {};
  size_t new_strides[CRV_MAX_RANK] = {};

  old_strides[rank - 1] = 1;
  new_strides[rank - 1] = 1;

  for (size_t i = rank - 1; i > 0; --i) {
    old_strides[i - 1] = old_dims[i] * old_strides[i];
    new_strides[i - 1] = new_dims[i] * old_strides[i];
  }

  float* x = tensor->data;
  float* y = tensor->swap;

  size_t indices[CRV_MAX_RANK] = {};

  for (size_t i = 0; i < item_count; ++i) {
    size_t tmp = i;
    for (size_t j = 0; j < rank; ++j) {
      indices[j] = tmp / old_strides[j];
      tmp %= old_strides[j];
    }

    tmp = indices[dim1];
    indices[dim1] = indices[dim2];
    indices[dim2] = tmp;

    size_t write_index = 0;
    for (size_t j = 0; j < rank; ++j) {
      write_index += indices[j] * new_strides[j];
    }

    y[write_index] = x[i];
  }

  tensor->dims[dim1] = new_dims[dim1];
  tensor->dims[dim2] = new_dims[dim2];

  tensor->data = y;
  tensor->swap = x;
}

void crv_tensor_permute(tensor_t* tensor, uint32_t* dims, uint32_t rank) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(tensor);
    assert(dims != NULL);  
    assert(rank > 0);
    assert(tensor->rank == rank);
  );

  size_t old_strides[CRV_MAX_RANK]; 
  size_t new_strides[CRV_MAX_RANK]; 

  size_t old_dims[CRV_MAX_RANK];
  size_t new_dims[CRV_MAX_RANK];

  for (size_t i = 0; i < rank; ++i) {
    old_dims[i] = tensor->dims[i];
    new_dims[i] = tensor->dims[dims[i]];
  }

  old_strides[rank - 1] = 1;
  new_strides[rank - 1] = 1;

  for (size_t i = rank - 1; i > 0; --i) {
    old_strides[i - 1] = old_strides[i] * old_dims[i];
    new_strides[i - 1] = new_strides[i] * new_dims[i];
  }

  size_t old_indices[CRV_MAX_RANK];
  size_t new_indices[CRV_MAX_RANK];
  size_t count = tensor->count;
  float* x = tensor->data;
  float* y = tensor->swap;

  for (size_t i = 0; i < count; ++i) {
    size_t tmp = i;
    for (size_t j = 0; j < rank; ++j) {
      old_indices[j] = tmp / old_strides[j];
      tmp %= old_strides[j];
    }

    for (size_t j = 0; j < rank; ++j) {
      new_indices[j] = old_indices[dims[j]];
    }

    size_t idx = 0;
    for (size_t j = 0; j < rank; ++j) {
      idx += new_indices[j] * new_strides[j];
    }

    y[idx] = x[i];
  }

  tensor->data = y;
  tensor->swap = x;

  for (size_t i = 0; i < rank; ++i) {
    tensor->dims[i] = new_dims[i];
  }
}

void crv_tensor_flip(tensor_t* tensor, uint32_t dim) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(tensor);
    assert(dim < tensor->rank);
    assert(tensor->dims[dim] > 1);
  );

  size_t rank = tensor->rank;
  size_t strides[CRV_MAX_RANK]; 
  crv_get_tensor_strides(tensor, &strides[0]);
  uint32_t* dims = tensor->dims; 

  size_t indices[CRV_MAX_RANK];
  size_t count = tensor->count;
  float* x = tensor->data;
  float* y = tensor->swap;

  for (size_t i = 0; i < count; ++i) {
    size_t tmp = i;
    for (size_t j = 0; j < rank; ++j) {
      indices[j] = tmp / strides[j];
      tmp %= strides[j];
    }

    indices[dim] = dims[dim] - indices[dim] - 1;

    size_t idx = 0;
    for (size_t j = 0; j < rank; ++j) {
      idx += indices[j] * strides[j];
    }

    y[idx] = x[i];
  }

  tensor->data = y;
  tensor->swap = x;
}

void crv_tensor_snake(tensor_t* tensor, tensor_t* alpha) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(tensor);
    crv_validate_tensor(alpha);
    assert(tensor->rank == 3);
    assert(alpha->rank == 2);
    assert(alpha->dims[1] == 1);
  );

  size_t batches = tensor->dims[0];
  size_t channels = tensor->dims[1];
  size_t len = tensor->dims[2];
  float* data = tensor->data;

  assert(channels == alpha->dims[0]);

  for (size_t b = 0; b < batches; ++b) {
    for (size_t ch = 0; ch < channels; ++ch) {
      for (size_t i = 0; i < len; ++i) {
        size_t idx = (b * channels * len) + (ch * len) + i;
        float x = data[idx];
        float s = alpha->data[ch];
        float value = sinf(s * x);
        data[idx] = x + value * value / (s + 1e-9f);
      }
    }
  }
}

void crv_tensor_leaky_relu(tensor_t* tensor, float alpha) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(tensor);
    assert(alpha >= 0.f && "Alpha value must be >= 0.");
  );

  size_t count = tensor->count;
  float* data = tensor->data;
  for (size_t i = 0; i < count; ++i) {
    data[i] = data[i] >= 0.f ? data[i] : data[i] * alpha;   
  }
}

void crv_tensor_sigmoid(tensor_t* tensor) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(tensor);
  );

  size_t count = tensor->count;
  float* data = tensor->data;
  for (size_t i = 0; i < count; ++i) {
    data[i] = 1.f / (1.f + expf(-data[i]));
  }
}

void crv_tensor_tanh(tensor_t* tensor) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(tensor);
  );

  size_t count = tensor->count;
  float* data = tensor->data;

  for (size_t i = 0; i < count; ++i) {
    data[i] = tanhf(data[i]);
  }
}

void crv_tensor_tmul(tensor_t* dest, tensor_t* src) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(dest);
    crv_validate_tensor(src);
    assert(src != NULL);
    assert(src != NULL);
    assert(src->rank == dest->rank);
    assert(src->count == dest->count);
    for (size_t i = 0; i < src->rank; ++i) {
      assert(src->dims[i] == dest->dims[i]);
    }
  );

  size_t count = src->count;
  for (size_t i = 0; i < count; ++i) {
    dest->data[i] *= src->data[i];
  }
}

void crv_tensor_tmul_last_dim(tensor_t* dest, tensor_t* src) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(dest);
    crv_validate_tensor(src);
    assert(src != NULL);
    assert(src != NULL);
    assert(src->dims[src->rank - 1] == dest->dims[dest->rank - 1]);
  );

  size_t src_len = src->dims[src->rank - 1];
  size_t dest_len = dest->dims[dest->rank - 1];

  for (size_t i = 0; i < dest_len; ++i) {
    dest->data[i] *= src->data[i % src_len];
  }
}

void crv_tensor_split(tensor_t* dest, tensor_t* src) {
  // TODO(luca): We will later expand on this. For now, we assume that the
  // input shape is [1, x, x], the split dim is 1 and the size is 2.

  CRV_DO_INTERNAL(
    crv_validate_tensor(dest);
    crv_validate_tensor(src);
    assert(src->rank == 3);
    assert(src->dims[0] == 1);
    assert(src->dims[1] % 2 == 0);
    assert(src->count % 2 == 0);
    assert(dest->cap >= src->count / 2);
  );

  float* x = src->data;
  float* y = dest->data;
  size_t count = src->count / 2;

  memcpy(y, &x[count], count * sizeof(float));

  src->dims[1] = src->dims[1] / 2;
  src->count = src->count / 2;

  dest->count = src->count;
  dest->rank = src->rank;

  for (size_t i = 0; i < dest->rank; ++i) {
    dest->dims[i] = src->dims[i];
  }
}

void crv_tensor_reshape(tensor_t* tensor, uint32_t* dims, uint32_t rank) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(tensor);
  );

  size_t count = 1;
  for (uint32_t i = 0; i < rank; ++i) {
    count *= dims[i]; 
    tensor->dims[i] = dims[i];  
  }

  assert(count == tensor->count);
  tensor->rank = rank;
}

void crv_tensor_conv1d(tensor_t* x, tensor_t* w, uint32_t stride, uint32_t dilation) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(x);
    crv_validate_tensor(w);
    assert(stride);
    assert(dilation);
    assert(x->rank == 3);
    assert(w->rank == 3);
  );
  size_t out_ch = w->dims[0];
  size_t in_ch = w->dims[1];
  size_t w_len = w->dims[2];

  size_t x_batches = x->dims[0];
  size_t x_in_ch = x->dims[1];
  size_t x_len = x->dims[2];
  assert(x_in_ch == in_ch);
  assert(!(w_len == 1 && dilation > 1));

  size_t eff_w_len = 1 + (w_len - 1) * dilation;
  size_t y_len = (x_len - eff_w_len) / stride + 1;
  assert(x_len >= eff_w_len);

  float* x_data = x->data;
  float* w_data = w->data;
  float* y_data = x->swap;

#ifdef CRV_IM2COL
  size_t strides[CRV_MAX_RANK];
  crv_get_tensor_strides(x, strides);

  size_t im2col_rows = in_ch * w_len;
  size_t im2col_cols = y_len;
  float* scratch = (float*)malloc(im2col_rows * im2col_cols * sizeof(float));
  assert(scratch != NULL);

  for (size_t b = 0; b < x_batches; ++b) {
    for (size_t m = 0; m < y_len; ++m) {
      for (size_t ic = 0; ic < in_ch; ++ic) {
        for (size_t k = 0; k < w_len; ++k) {
          size_t x_idx = m * stride + k * dilation;
          size_t x_read_index = b * strides[0] + ic * strides[1] + x_idx;
          size_t row = ic * w_len + k;
          scratch[row * im2col_cols + m] = x_data[x_read_index];
        }
      }
    }

    float* y_batch = y_data + b * out_ch * y_len;
    for (size_t i = 0; i < out_ch * y_len; ++i) {
      y_batch[i] = 0.0f;
    }

    //cblas_sgemm(
    //  CblasRowMajor, CblasNoTrans, CblasNoTrans,
    //  out_ch, y_len, im2col_rows,
    //  1.0f, w_data, im2col_rows, scratch, y_len,
    //  0.0f, y_batch, y_len
    //);

    crv_mm(y_batch, scratch, w_data, out_ch, im2col_rows, y_len);
  }

  free(scratch);
#else
  for (size_t b = 0; b < x_batches; ++b) {
    for (size_t oc = 0; oc < out_ch; ++oc) {
      for (size_t m = 0; m < y_len; ++m) {
        float sum = 0;
        for (size_t ic = 0; ic < in_ch; ++ic) {
          for (size_t k = 0; k < w_len; ++k) {
            size_t x_idx = m * stride + k * dilation;
            size_t x_read_index = (b * in_ch * x_len) + (ic * x_len) + x_idx;
            size_t w_read_index = (oc * in_ch * w_len) + (ic * w_len) + k;
            sum += x_data[x_read_index] * w_data[w_read_index];
          }
        }

        size_t write_index = (b * out_ch * y_len) + (oc * y_len) + m;
        y_data[write_index] = sum;
      }
    }
  }
#endif
  x->dims[1] = out_ch;
  x->dims[2] = y_len;
  x->count = x->dims[0] * x->dims[1] * x->dims[2];
  assert(x->count <= x->cap);

  x->data = y_data;
  x->swap = x_data;
}

void crv_tensor_conv_transpose1d(tensor_t* x, tensor_t* w, uint32_t stride, uint32_t dilation) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(x);
    crv_validate_tensor(w);
    assert(stride);
    assert(dilation);
    assert(x->rank == 3);
    assert(w->rank == 3);
  );

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

//#ifdef CRV_IM2COL
//  assert(batches == 1);
//
//  size_t rows = y_len;
//  size_t cols = w_len;
//  size_t N = rows * cols;
//
//  float* im2col = (float*)malloc(N * sizeof(float));
//  memset(im2col, 0, N * sizeof(float));
//    
//  for (size_t b = 0; b < batches; ++b) {
//    for (size_t ic = 0; ic < in_ch; ++ic) {
//      for () {
//
//      }
//    }
//  }
//
//  free(im2col);
//#else
  for (size_t b = 0; b < batches; ++b) {
    for (size_t ic = 0; ic < x_in_ch; ++ic) {
      for (size_t i = 0; i < x_len; ++i) {
        for (size_t oc = 0; oc < out_ch; ++oc) {
          for (size_t k = 0; k < w_len; ++k) {
            size_t out_idx = i * stride + k * dilation;
            size_t x_idx = (b * x_in_ch * x_len) + (ic * x_len) + i;
            size_t y_idx = (b * out_ch * y_len) + (oc * y_len) + out_idx;
            size_t w_idx = (ic * out_ch * w_len) + (oc * w_len) + k;
            y_data[y_idx] += x_data[x_idx] * w_data[w_idx];
          }
        }
      }
    }
  }
//#endif

  x->data = y_data;
  x->swap = x_data;
}

void crv_tensor_rfft(tensor_t* input) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(input);
    assert(input->rank >= 1);
  );

  size_t rank = input->rank;
  size_t batches = 1;
  for (size_t i = 0; i < rank - 1; ++i) {
    batches *= input->dims[i];
  }

  float* x = input->data;
  float* y = input->swap;
  size_t n = input->dims[input->rank - 1];
  size_t len = n / 2 + 1;
  size_t count = batches * len * 2;
  assert(count <= input->cap);

  for (size_t b = 0; b < batches; ++b) {
    for (size_t k = 0; k < len; ++k) {
      float real = 0.f;
      float imag = 0.f;

      for (size_t t = 0; t < n; ++t) {
        float angle = -2.f * (float)M_PI * k * t / n;
        float val = x[b * n + t];
        real += val * cosf(angle);
        imag += val * sinf(angle);
      }

      y[b * len * 2 + k * 2] = real;
      y[b * len * 2 + k * 2 + 1] = imag;
    }
  }

  input->data = y;
  input->swap = x;
  input->rank = rank + 1;
  input->dims[rank - 1] = len;
  input->dims[rank] = 2;
  input->count = count;
}

void crv_tensor_irfft(tensor_t* input) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(input);
    assert(input->rank > 1);
    assert(input->dims[input->rank - 1] == 2);
  );

  size_t rank = input->rank;
  size_t batches = 1;
  for (size_t i = 0; i < rank - 2; ++i) {
    batches *= input->dims[i];
  }

  float* x = input->data;
  float* y = input->swap;
  size_t len = input->dims[input->rank - 2];
  size_t n = 2 * (len - 1);
  size_t count = n * batches;
  assert(count <= input->cap);

  for (size_t b = 0; b < batches; ++b) {
    for (size_t t = 0; t < n; ++t) {
      float sum = 0.f;

      for (int k = 0; k < len; ++k) {
        float angle = 2.f * (float)M_PI * k * t / n;
        float r = x[b * len * 2 + k * 2];
        float i = x[b * len * 2 + k * 2 + 1];
        sum += r * cosf(angle) - i * sinf(angle);
      }

      y[b * n + t] = sum / n;
    }
  }

  input->data = y;
  input->swap = x;
  input->rank = rank - 1;
  input->dims[rank - 2] = n;
  input->count = count;
}

float crv_tensor_l1_norm(tensor_t* a, tensor_t* b) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(a);
    crv_validate_tensor(b);
    assert(a->rank == b->rank);
    assert(a->dims[0] == b->dims[0]);
    assert(a->dims[1] == b->dims[1]);
    assert(a->dims[2] == b->dims[2]);
    assert(a->count == b->count);
  );

  size_t count = a->count;
  float diff = 0;
  for (size_t i = 0; i < count; ++i) {
    float result = fabs(a->data[i] - b->data[i]);
    diff += result;
  }

  return diff;
}

float crv_tensor_mae(tensor_t* a, tensor_t* b) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(a);
    crv_validate_tensor(b);
    assert(a->rank == b->rank);
    assert(a->dims[0] == b->dims[0]);
    assert(a->dims[1] == b->dims[1]);
    assert(a->dims[2] == b->dims[2]);
    assert(a->count == b->count);
  );

  float diff = crv_tensor_l1_norm(a, b);
  return diff / (float)a->count;
}

float crv_tensor_maxae(tensor_t* a, tensor_t* b) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(a);
    crv_validate_tensor(b);
    assert(a->rank == b->rank);
    assert(a->dims[0] == b->dims[0]);
    assert(a->dims[1] == b->dims[1]);
    assert(a->dims[2] == b->dims[2]);
    assert(a->count == b->count);
  );

  assert(a->rank == b->rank);
  assert(a->dims[0] == b->dims[0]);
  assert(a->dims[1] == b->dims[1]);
  assert(a->dims[2] == b->dims[2]);
  assert(a->count == b->count);
  assert(a->count != 0);

  float max_diff = 0;
  size_t count = a->count;
  for (size_t i = 0; i < count; ++i) {
    float result = fabs(a->data[i] - b->data[i]);
    if (result > max_diff) {
      max_diff = result;
    }
  }

  return max_diff;
}

void crv_tensor_print_error_stats(tensor_t* a, tensor_t* b) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(a);
    crv_validate_tensor(b);
    assert(a->rank == b->rank);
    assert(a->dims[0] == b->dims[0]);
    assert(a->dims[1] == b->dims[1]);
    assert(a->dims[2] == b->dims[2]);
    assert(a->count == b->count);
  );

  float l1_norm = crv_tensor_l1_norm(a, b);
  float mae = crv_tensor_mae(a, b);
  float maxae = crv_tensor_maxae(a, b);

  printf("L1 Norm Error:       %.12f\n", l1_norm);
  printf("Max Absolute Error:  %.12f\n", maxae);
  printf("Mean Absolute Error: %.12f\n", mae);
}

void crv_tensor_print_shape(tensor_t* tensor) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(tensor);
  );

  size_t rank = tensor->rank;

  printf("%s shape: [", tensor->name); 

  for (uint32_t i = 0; i < rank - 1; ++i) {
    printf("%u, ", tensor->dims[i]);
  }

  printf("%u]\n", tensor->dims[rank - 1]);
}

void crv_tensor_print_data(tensor_t* tensor) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(tensor);
  );

  // TODO(luca): Add nicer tensor printing.
  printf("Content of tensor: %s\n", tensor->name); 
  for (size_t i = 0; i < tensor->count; ++i) {
    printf("%.0f, ", tensor->data[i]);
  }
  printf("\n");
}

void crv_tensor_print(tensor_t* tensor) {
  CRV_DO_INTERNAL(
    crv_validate_tensor(tensor);
  );

  crv_tensor_print_shape(tensor);
  crv_tensor_print_data(tensor);
}

#endif // CRV_IMPLEMENTATION
