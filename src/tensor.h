#ifndef TENSOR_H
#define TENSOR_H

#define TENSOR_MAX_RANK 6
#define TENSOR_AUTO_CAP 0

// TODO(luca): Use size_t for some items such as count and cap.
typedef struct {
  uint32_t cap;
  uint32_t count;
  uint32_t rank;
  uint32_t dims[TENSOR_MAX_RANK];
  float* data;
  float* swap;
  char* name;
} tensor_t;

typedef struct {
  tensor_t** tensors;
  size_t count;
} tensor_list_t;

// NOTE(luca): Example:
// tensor_t* t = tensor_create(U32_TPL(1, 8, 16), TENSOR_AUTO_CAP);
#define U32_TPL(...) \
  (uint32_t[]) {__VA_ARGS__}, sizeof((uint32_t[]) {__VA_ARGS__}) / sizeof(uint32_t)

static void tensor_validate(tensor_t* tensor);
static void tensor_get_strides(tensor_t* tensor, size_t* strides);
static tensor_t* tensor_create(arena_t* arena, uint32_t* dims, uint32_t rank, uint32_t capacity);
static tensor_t* tensor_find_in_list(tensor_list_t* list, char* name);
static tensor_list_t* tensor_load_from_blob(arena_t* arena, char* path);
static tensor_t* tensor_load_from_stream(arena_t* arena, FILE* file, uint32_t min_capacity);
static tensor_t* tensor_load_from_file(arena_t* arena, char* path, uint32_t min_capacity);
static void tensor_save_to_file(tensor_t* tensor, char* path);
static void tensor_fill(tensor_t* restrict tensor, float val);
static void tensor_mul(tensor_t* restrict tensor, float mul);
static void tensor_add(tensor_t* restrict tensor, float add);
static void tensor_tadd(tensor_t* restrict dest, tensor_t* src);
static void tensor_arange(tensor_t* restrict tensor, float start, float step);
static void tensor_cat(tensor_t* restrict dest, tensor_t* restrict src, uint32_t dim);
static void tensor_pad(tensor_t* restrict tensor, size_t left_pad);
static void tensor_trunc(tensor_t* restrict tensor, uint32_t left_trunc, uint32_t right_trunc);
static void tensor_copy(tensor_t* restrict dest, tensor_t* restrict src);
static void tensor_squeeze(tensor_t* restrict tensor, uint32_t dim);
static void tensor_unsqueeze(tensor_t* restrict tensor, uint32_t dim);
static void tensor_transpose(tensor_t* restrict tensor, uint32_t dim1, uint32_t dim2);
static void tensor_permute(tensor_t* restrict tensor, uint32_t* restrict dims, uint32_t rank);
static void tensor_flip(tensor_t* restrict tensor, uint32_t dim);
static void tensor_snake(tensor_t* restrict tensor, tensor_t* restrict alpha);
static void tensor_leaky_relu(tensor_t* restrict tensor, float alpha);
static void tensor_sigmoid(tensor_t* restrict tensor);
static void tensor_tanh(tensor_t* restrict tensor);
static void tensor_tmul(tensor_t* restrict a, tensor_t* restrict b);
static void tensor_split(tensor_t* restrict dest, tensor_t* restrict src);
static void tensor_reshape(tensor_t* restrict tensor, uint32_t* restrict dims, uint32_t rank);
static void tensor_conv1d(tensor_t* restrict x, tensor_t* restrict w, uint32_t stride, uint32_t dilation);
static void tensor_conv_transpose1d(tensor_t* restrict x, tensor_t* restrict w, uint32_t stride, uint32_t dilation);
static float tensor_l1_norm(tensor_t* restrict a, tensor_t* restrict b);
static float tensor_mae(tensor_t* restrict a, tensor_t* restrict b);
static float tensor_maxae(tensor_t* restrict a, tensor_t* restrict b);
static void tensor_print_error_stats(tensor_t* restrict a, tensor_t* restrict b);
static void tensor_print_shape(tensor_t* restrict tensor);
static void tensor_print_data(tensor_t* restrict tensor);
static void tensor_print(tensor_t* restrict tensor);

void tensor_validate(tensor_t* tensor) {
  assert(tensor != NULL                   && "Tensor is NULL.");
  assert(tensor->data != NULL             && "Tensor data is NULL.");
  assert(tensor->swap != NULL             && "Tensor swap is NULL.");
  assert(tensor->rank > 0                 && "Tensor rank is 0.");
  assert(tensor->rank <= TENSOR_MAX_RANK  && "Tensor rank greater than TENSOR_MAX_RANK.");
  assert(tensor->count <= tensor->cap     && "Tensor count exceeds capacity.");
  assert(tensor->cap > 0                  && "Tensor capacity is 0.");
  assert(tensor->count > 0                && "Tensor count is 0.");
  
  size_t count = 1;
  for (size_t i = 0; i < tensor->rank; ++i) {
    count *= tensor->dims[i];
  }

  assert(count == tensor->count);
}

void tensor_get_strides(tensor_t* tensor, size_t* strides) {
  DO_INTERNAL(
    tensor_validate(tensor);
    assert(strides != NULL);
  );

  size_t rank = tensor->rank;
  uint32_t* dims = tensor->dims;
  strides[rank - 1] = 1;

  for (size_t i = rank - 1; i > 0; --i) {
    strides[i - 1] = strides[i] * dims[i]; 
  }
}

tensor_t* tensor_create(arena_t* arena, uint32_t* dims, uint32_t rank, uint32_t capacity) {
  assert(rank > 0);
  assert(dims != NULL);

  tensor_t* tensor = arena_alloc(arena, sizeof(tensor_t));
  assert(tensor);

  tensor->rank = rank;

  tensor->count = 1;

  for (uint32_t i = 0; i < rank; ++i) {
    assert(dims[i]);
    tensor->dims[i] = dims[i];
    tensor->count *= dims[i];
  }

  if (capacity == TENSOR_AUTO_CAP) {
    tensor->cap = tensor->count;    
  } else {
    tensor->cap = capacity;
    assert(tensor->count <= capacity);
  }

  tensor->data = arena_alloc(arena, tensor->cap * sizeof(float));
  assert(tensor->data);
  tensor->swap = arena_alloc(arena, tensor->cap * sizeof(float));
  assert(tensor->swap);

  return tensor;
}

void tensor_init(tensor_t* tensor, uint32_t* dims, uint32_t rank) {
  DO_INTERNAL(
    tensor_validate(tensor);
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

tensor_t* tensor_find_in_list(tensor_list_t* list, char* name) {
  for (size_t i = 0; i < list->count; ++i) {
    if (strcmp(list->tensors[i]->name, name) == 0) {
      return list->tensors[i];
    }
  }

  return NULL;
}

tensor_t* tensor_load_from_stream(arena_t* arena, FILE* file, uint32_t min_capacity) {
  // FORMAT [name_len (uint32_t)] [name (char * name_len)]
  // [rank (uint32_t)] [dims (uint32_t * rank)]
  // [item_count (uint32_t)] [data (float * item_count)]

  assert(file != NULL);

  uint32_t name_len;
  int result = fread(&name_len, sizeof(uint32_t), 1, file);
  CHECK_GOTO(result == 1, error, "Error.\n");

  char* name = arena_alloc(arena, name_len);
  result = fread(name, sizeof(char), name_len, file);
  CHECK_GOTO(result == (int)name_len, error, "Error.\n");

  uint32_t rank;
  result = fread(&rank, sizeof(uint32_t), 1, file);
  CHECK_GOTO(result == 1, error, "Error.\n");

  uint32_t dims[TENSOR_MAX_RANK];
  result = fread(dims, sizeof(uint32_t), rank, file);
  CHECK_GOTO(result == (int)rank, error, "Error.\n");

  uint32_t item_count;
  result = fread(&item_count, sizeof(uint32_t), 1, file);
  CHECK_GOTO(result == 1, error, "Error\n");

  uint32_t capacity = item_count < min_capacity ? min_capacity : item_count;
  tensor_t* tensor = tensor_create(arena, dims, rank, capacity);
  CHECK_GOTO(tensor, error, "Failed to allocate memory for tensor.\n");

  result = fread(tensor->data, sizeof(float), item_count, file);
  CHECK_GOTO(result == (int)item_count, error, "Error.\n");

  tensor->name = name;

  return tensor;

error:
  return NULL;
}

tensor_list_t* tensor_load_from_blob(arena_t* arena, char* path) {
  assert(path);

  // TODO(luca): Add file.
  FILE* file = fopen(path, "rb");
  CHECK_GOTO(file, error, "Failed to open file: %s.\n", path);

  uint32_t count;
  int result = fread(&count, sizeof(uint32_t), 1, file);
  CHECK_GOTO(result == 1, error, "Failed to read from file.");

  tensor_list_t* list = arena_alloc(arena, sizeof(tensor_list_t*));
  list->tensors = arena_alloc(arena, count * sizeof(tensor_t*));
  list->count = count;

  for (int i = 0; i < count; ++i) {
    list->tensors[i] = tensor_load_from_stream(arena, file, TENSOR_AUTO_CAP);
  }

  fclose(file);
  return list;
    
error:
  fclose(file);
  return NULL;
}

tensor_t* tensor_load_from_file(arena_t* arena, char* path, uint32_t min_capacity) {
  // FORMAT [name_len (uint32_t)] [name (char * name_len)]
  // [rank (uint32_t)] [dims (uint32_t * rank)]
  // [item_count (uint32_t)] [data (float * item_count)]

  assert(path != NULL);

  // TODO(luca): See if it is possible to lock file while reading/writing.
  // TODO(luca): Check that the size of the reads matches the expected size.
  // TODO(luca): We still want to use size_t.
  // TODO(luca): We want to ensure that the data is packed as uint and not int
  // in the Python script.
  // TODO(luca): Add better error logging.
  FILE* file = fopen(path, "rb");
  CHECK_GOTO(file, error, "Failed to open file: %s.\n", path);

  uint32_t name_len;
  int result = fread(&name_len, sizeof(uint32_t), 1, file);
  CHECK_GOTO(result, error, "Error.\n");

  char* name = arena_alloc(arena, name_len);
  result = fread(name, sizeof(char), name_len, file);
  CHECK_GOTO(result, error, "Error.\n");

  uint32_t rank;
  result = fread(&rank, sizeof(uint32_t), 1, file);
  CHECK_GOTO(result, error, "Error.\n");

  uint32_t dims[TENSOR_MAX_RANK];
  result = fread(dims, sizeof(uint32_t), rank, file);
  CHECK_GOTO(result, error, "Error.\n");

  uint32_t item_count;
  result = fread(&item_count, sizeof(uint32_t), 1, file);
  CHECK_GOTO(result, error, "Error\n");

  uint32_t capacity = item_count < min_capacity ? min_capacity : item_count;
  tensor_t* tensor = tensor_create(arena, dims, rank, capacity);
  CHECK_GOTO(tensor, error, "Failed to allocate memory for tensor.\n");

  result = fread(tensor->data, sizeof(float), item_count, file);
  CHECK_GOTO(result, error, "Error.\n");
  fclose(file);

  tensor->name = name;

  return tensor;

error:
  fclose(file);
  return NULL;
}

void tensor_save_to_file(tensor_t* tensor, char* path) {
  // FORMAT [name_len (uint32_t)] [name (char * name_len)]
  // [rank (uint32_t)] [dims (uint32_t * rank)]
  // [item_count (uint32_t)] [data (float * item_count)]

  DO_INTERNAL(
    tensor_validate(tensor);
    assert(path != NULL);
  );

  FILE* file = fopen(path, "wb");
  CHECK_GOTO(file, error, "Failed to open file: %s.\n", path);

  uint32_t name_len = strlen(tensor->name) + 1;

  int result;
  result = fwrite(&name_len, sizeof(uint32_t), 1, file);
  CHECK_GOTO(result, error, "Failed to write name_len to file %s.\n", path);

  result = fwrite(tensor->name, sizeof(char), name_len, file);
  CHECK_GOTO(result, error, "Failed to write name to file %s.\n", path);

  result = fwrite(&tensor->rank, sizeof(uint32_t), 1, file);
  CHECK_GOTO(result, error, "Failed to write rank to file %s.\n", path);

  result = fwrite(tensor->dims, sizeof(uint32_t), tensor->rank, file);
  CHECK_GOTO(result, error, "Failed to write rank to file %s.\n", path);

  result = fwrite(&tensor->count, sizeof(uint32_t), 1, file);
  CHECK_GOTO(result, error, "Failed to write item_count to file %s.\n", path);

  result = fwrite(tensor->data, sizeof(float), tensor->count, file);
  CHECK_GOTO(result, error, "Failed to write data to file %s.\n", path);

  fclose(file);

error:
  fclose(file);
}

void tensor_fill(tensor_t* restrict tensor, float val) {
  DO_INTERNAL(
    tensor_validate(tensor);
  );

  for (size_t i = 0; i < tensor->count; ++i) {
    tensor->data[i] = val;
  }
}

void tensor_mul(tensor_t* restrict tensor, float mul) {
  DO_INTERNAL(
    tensor_validate(tensor);
  );

  for (uint32_t i = 0; i < tensor->count; ++i) {
    tensor->data[i] *= mul;
  }
}

void tensor_add(tensor_t* restrict tensor, float add) {
  DO_INTERNAL(
    tensor_validate(tensor);
  );

  for (uint32_t i = 0; i < tensor->count; ++i) {
    tensor->data[i] += add;
  }
}

void tensor_tadd(tensor_t* restrict dest, tensor_t* src) {
  DO_INTERNAL(
    tensor_validate(src);
    tensor_validate(dest);
    assert(src->rank == dest->rank && "Tensor rank must match.");
    assert(src->count == dest->count && "Tensor item count must match.");
  );

  for (size_t i = 0; i < dest->count; ++i) {
    dest->data[i] += src->data[i];
  }
}

void tensor_arange(tensor_t* restrict tensor, float start, float step) {
  DO_INTERNAL(
    tensor_validate(tensor);
  );

  for (size_t i = 0; i < tensor->count; ++i) {
    tensor->data[i] = (float)i * step + start;
  }
}

void tensor_cat(tensor_t* restrict dest, tensor_t* restrict src, uint32_t dim) {
  DO_INTERNAL(
    tensor_validate(src);
    tensor_validate(dest);
    assert(src->rank == dest->rank);
    assert(dim < dest->rank && "Dimension must be smaller than rank.");

    for (uint32_t i = 0; i < dest->rank; ++i) {
      if (i != dim) {
        assert(dest->dims[i] == src->dims[i]);
      }
    }
  );

  size_t rank = dest->rank;
  size_t a_cpy_size = dest->dims[dim];
  size_t b_cpy_size = src->dims[dim];

  for (uint32_t i = dim + 1; i < rank; ++i) {
    a_cpy_size *= dest->dims[i];
    b_cpy_size *= src->dims[i];
  }

  float* out = dest->swap;
  float* a = dest->data;
  float* b = src->data;

  size_t copied_count = 0;
  size_t total_items = dest->count + src->count;
  assert(total_items <= dest->cap && "Total number of items will exceed tensor capacity.");

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

void tensor_pad(tensor_t* restrict tensor, size_t left_pad) {
  // TODO(luca): Not a complete implementation of a tensor padding algorithm as
  // this only does left padding.

  DO_INTERNAL(
    tensor_validate(tensor);
  );

  float* x = tensor->data;
  float* y = tensor->swap;

  size_t rank = tensor->rank;
  size_t item_count = tensor->dims[tensor->rank - 1];

  size_t total_dims = 1;
  for (uint32_t i = 0; i < rank - 1; ++i) {
    total_dims *= tensor->dims[i];
  }

  assert(total_dims * (left_pad + item_count) <= tensor->cap);

  for (size_t i = 0; i < total_dims; ++i) {
    for (size_t j = 0; j < left_pad; ++j) {
      y[i * item_count + i * left_pad + j] = 0.f;
    }

    memcpy(
      &y[i * item_count + i * left_pad + left_pad],
      &x[i * item_count],
      item_count * sizeof(float)
    );
  }

  tensor->data = y;
  tensor->swap = x;
  tensor->count = total_dims * (item_count + left_pad);
  tensor->dims[rank - 1] += left_pad;
}

void tensor_trunc(tensor_t* restrict tensor, uint32_t left_trunc, uint32_t right_trunc) {
  // TODO(luca): Only implements left and right truncation.

  DO_INTERNAL(
    tensor_validate(tensor);
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

void tensor_copy(tensor_t* restrict dest, tensor_t* restrict src) {
  DO_INTERNAL(
    tensor_validate(src);
    tensor_validate(dest);
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

void tensor_squeeze(tensor_t* restrict tensor, uint32_t dim) {
  DO_INTERNAL(
    tensor_validate(tensor);
    assert(dim <= tensor->rank);
    assert(tensor->dims[dim] == 1);
  );

  size_t rank = tensor->rank;
  for (size_t i = dim; i < rank - 1; --i) {
    tensor->dims[i] = tensor->dims[i + 1];
  }

  tensor->rank = rank - 1;
}

void tensor_unsqueeze(tensor_t* restrict tensor, uint32_t dim) {
  DO_INTERNAL(
    tensor_validate(tensor);
    assert(dim <= tensor->rank);
  );

  size_t rank = tensor->rank;
  for (size_t i = rank; i > dim; --i) {
    tensor->dims[i] = tensor->dims[i - 1];
  }

  tensor->dims[dim] = 1;
  tensor->rank = rank + 1;
}

void tensor_transpose(tensor_t* restrict tensor, uint32_t dim1, uint32_t dim2) {
  DO_INTERNAL(
    tensor_validate(tensor);
    assert(dim1 < tensor->rank);
    assert(dim2 < tensor->rank);
  );

  size_t rank = tensor->rank;
  size_t item_count = tensor->count;

  size_t old_dims[TENSOR_MAX_RANK] = {};
  size_t new_dims[TENSOR_MAX_RANK] = {};

  for (size_t i = 0; i < rank; ++i) {
    old_dims[i] = tensor->dims[i];
    new_dims[i] = tensor->dims[i];
  }

  new_dims[dim1] = old_dims[dim2];
  new_dims[dim2] = old_dims[dim1];

  size_t old_strides[TENSOR_MAX_RANK] = {};
  size_t new_strides[TENSOR_MAX_RANK] = {};

  old_strides[rank - 1] = 1;
  new_strides[rank - 1] = 1;

  for (size_t i = rank - 1; i > 0; --i) {
    old_strides[i - 1] = old_dims[i] * old_strides[i];
    new_strides[i - 1] = new_dims[i] * old_strides[i];
  }

  float* x = tensor->data;
  float* y = tensor->swap;

  size_t indices[TENSOR_MAX_RANK] = {};

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

void tensor_permute(tensor_t* restrict tensor, uint32_t* restrict dims, uint32_t rank) {
  DO_INTERNAL(
    tensor_validate(tensor);
    assert(dims != NULL);  
    assert(rank > 0);
    assert(tensor->rank == rank);
  );

  size_t old_strides[TENSOR_MAX_RANK]; 
  size_t new_strides[TENSOR_MAX_RANK]; 

  size_t old_dims[TENSOR_MAX_RANK];
  size_t new_dims[TENSOR_MAX_RANK];

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

  size_t old_indices[TENSOR_MAX_RANK];
  size_t new_indices[TENSOR_MAX_RANK];
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

void tensor_flip(tensor_t* restrict tensor, uint32_t dim) {
  DO_INTERNAL(
    tensor_validate(tensor);
    assert(dim < tensor->rank);
    assert(tensor->dims[dim] > 1);
  );

  size_t rank = tensor->rank;
  size_t strides[TENSOR_MAX_RANK]; 
  tensor_get_strides(tensor, &strides[0]);
  uint32_t* dims = tensor->dims; 

  size_t indices[TENSOR_MAX_RANK];
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

void tensor_snake(tensor_t* restrict tensor, tensor_t* restrict alpha) {
  DO_INTERNAL(
    tensor_validate(tensor);
    tensor_validate(alpha);
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

void tensor_leaky_relu(tensor_t* restrict tensor, float alpha) {
  DO_INTERNAL(
    tensor_validate(tensor);
    assert(alpha >= 0.f && "Alpha value must be >= 0.");
  );

  size_t count = tensor->count;
  float* data = tensor->data;
  for (size_t i = 0; i < count; ++i) {
    data[i] = data[i] >= 0.f ? data[i] : data[i] * alpha;   
  }
}

void tensor_sigmoid(tensor_t* restrict tensor) {
  DO_INTERNAL(
    tensor_validate(tensor);
  );

  size_t count = tensor->count;
  float* data = tensor->data;
  for (size_t i = 0; i < count; ++i) {
    data[i] = 1.f / (1.f + expf(-data[i]));
  }
}

void tensor_tanh(tensor_t* restrict tensor) {
  DO_INTERNAL(
    tensor_validate(tensor);
  );

  size_t count = tensor->count;
  float* data = tensor->data;

  for (size_t i = 0; i < count; ++i) {
    data[i] = tanhf(data[i]);
  }
}

void tensor_tmul(tensor_t* restrict a, tensor_t* restrict b) {
  assert(a && "Tensor a is NULL.");
  assert(b && "Tensor a is NULL.");
  assert(a->rank == b->rank && "Tensor rank must match.");
  assert(a->count == b->count && "Tensor item count must match.");

  for (size_t i = 0; i < a->count; ++i) {
    a->data[i] *= b->data[i];
  }
}

void tensor_split(tensor_t* restrict dest, tensor_t* restrict src) {
  // TODO(luca): We will later expand on this. For now, we assume that the
  // input shape is [1, x, x], the split dim is 1 and the size is 2.

  DO_INTERNAL(
    tensor_validate(dest);
    tensor_validate(src);
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

void tensor_reshape(tensor_t* restrict tensor, uint32_t* restrict dims, uint32_t rank) {
  DO_INTERNAL(
    tensor_validate(tensor);
  );

  size_t count = 1;
  for (uint32_t i = 0; i < rank; ++i) {
    count *= dims[i]; 
    tensor->dims[i] = dims[i];  
  }

  assert(count == tensor->count);
  tensor->rank = rank;
}

void tensor_conv1d(tensor_t* restrict x, tensor_t* restrict w, uint32_t stride, uint32_t dilation) {
  DO_INTERNAL(
    tensor_validate(x);
    tensor_validate(w);
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

  x->dims[1] = out_ch;
  x->dims[2] = y_len;
  x->count = x->dims[0] * x->dims[1] * x->dims[2];

  assert(x->count <= x->cap);

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

  x->data = y_data;
  x->swap = x_data;
}

void tensor_conv_transpose1d(tensor_t* restrict x, tensor_t* restrict w, uint32_t stride, uint32_t dilation) {
  DO_INTERNAL(
    tensor_validate(x);
    tensor_validate(w);
    assert(stride);
    assert(dilation);
    assert(x->rank == 3);
    assert(w->rank == 3);
  );

  size_t in_ch = w->dims[0];
  size_t out_ch = w->dims[1];
  size_t w_len = w->dims[2];

  size_t x_batches = x->dims[0];
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

  for (size_t i = 0; i < x->count; ++i) {
    y_data[i] = 0;
  }

  for (size_t b = 0; b < x_batches; ++b) {
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

  x->data = y_data;
  x->swap = x_data;
}

float tensor_l1_norm(tensor_t* restrict a, tensor_t* restrict b) {
  DO_INTERNAL(
    tensor_validate(a);
    tensor_validate(b);
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

float tensor_mae(tensor_t* restrict a, tensor_t* restrict b) {
  DO_INTERNAL(
    tensor_validate(a);
    tensor_validate(b);
    assert(a->rank == b->rank);
    assert(a->dims[0] == b->dims[0]);
    assert(a->dims[1] == b->dims[1]);
    assert(a->dims[2] == b->dims[2]);
    assert(a->count == b->count);
  );

  float diff = tensor_l1_norm(a, b);
  return diff / (float)a->count;
}

float tensor_maxae(tensor_t* restrict a, tensor_t* restrict b) {
  DO_INTERNAL(
    tensor_validate(a);
    tensor_validate(b);
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

void tensor_print_error_stats(tensor_t* restrict a, tensor_t* restrict b) {
  DO_INTERNAL(
    tensor_validate(a);
    tensor_validate(b);
    assert(a->rank == b->rank);
    assert(a->dims[0] == b->dims[0]);
    assert(a->dims[1] == b->dims[1]);
    assert(a->dims[2] == b->dims[2]);
    assert(a->count == b->count);
  );

  float l1_norm = tensor_l1_norm(a, b);
  float mae = tensor_mae(a, b);
  float maxae = tensor_maxae(a, b);

  printf("L1 Norm Error:       %.12f\n", l1_norm);
  printf("Max Absolute Error:  %.12f\n", maxae);
  printf("Mean Absolute Error: %.12f\n", mae);
}

void tensor_print_shape(tensor_t* restrict tensor) {
  DO_INTERNAL(
    tensor_validate(tensor);
  );

  size_t rank = tensor->rank;

  printf("%s shape: [", tensor->name); 

  for (uint32_t i = 0; i < rank - 1; ++i) {
    printf("%u, ", tensor->dims[i]);
  }

  printf("%u]\n", tensor->dims[rank - 1]);
}

void tensor_print_data(tensor_t* restrict tensor) {
  DO_INTERNAL(
    tensor_validate(tensor);
  );

  // TODO(luca): Add nicer tensor printing.
  printf("Content of tensor: %s\n", tensor->name); 
  for (size_t i = 0; i < tensor->count; ++i) {
    printf("%.0f, ", tensor->data[i]);
  }
  printf("\n");
}

void tensor_print(tensor_t* restrict tensor) {
  DO_INTERNAL(
    tensor_validate(tensor);
  );

  tensor_print_shape(tensor);
  tensor_print_data(tensor);
}

#undef TENSOR_MAX_RANK
#endif // TENSOR_H
