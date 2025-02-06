#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>

typedef int s32;
typedef unsigned int u32;
typedef float f32;
typedef double f64;

typedef struct {
  f32* data;
  s32 len;
  s32 size;
} Tensor;

typedef struct {
  Tensor alpha;  
} Snake;

typedef struct {
   
} DilatedUnit;

Tensor CreateTensor(s32 size) {
  Tensor tensor = {
    .data = (f32*)calloc(size, sizeof(f32)),
    .len = size,
    .size = size 
  };

  return tensor;
}

Snake CreateSnake(s32 dimension_count) {
  Snake snake = {
    .alpha = CreateTensor(dimension_count)
  };

  // TODO(luca): We most likely will not need to do this later as we will be
  // loading the alpha values from our model directly.
  for (s32 i = 0; i < dimension_count; ++i) {
    snake.alpha.data[i] = 1.f;
  }

  return snake;
}

void DoLeakyRelu(Tensor* input, f32 alpha) {
  for (s32 i = 0; i < input->len; ++i) {
    if (input->data[i] < 0) {
      input->data[i] = input->data[i] * alpha;
    }
  }
}

void DoConv1d(Tensor* input, Tensor* kernel, s32 stride, s32 dilation) {
  // TODO(luca): Add padding.

  assert(input->len && "We convolved too much.");

  s32 output_len = (input->len - dilation * (kernel->len - 1) - 1) / stride + 1;
  s32 i, j;
  f32 result;

  for (i = 0; i < output_len; i += stride) {
    result = 0;
    for (j = 0; j < kernel->len; ++j) {
      result += input->data[i + j * dilation] * kernel->data[j];
    }

    input->data[i] = result;
  }

  //input->len = output_len;
}

void DoSnake(Tensor* input, Snake* snake) {
  f32 eps = 1e-9f;
  s32 dim_count = snake->alpha.len;
  f32 alpha, alpha_inv;
  s32 dim_index, i;
  f32 val;
  
  for (dim_index = 0; dim_index < dim_count; ++dim_index) {
    alpha = snake->alpha.data[dim_index];
    alpha_inv = 1.f / (alpha + eps);

    for (i = 0; i < input->len; ++i) {
      val = input->data[i];
      input->data[i] = val + alpha_inv * pow(sinf(alpha + val), 2.f);
    }
  }
}

void DoDilatedUnit(Tensor* input, Tensor* kernel, s32 stride, s32 dilation) {
  DoLeakyRelu(input, 0.2f);
  DoConv1d(input, kernel, stride, dilation);
  DoLeakyRelu(input, 0.2f);
  DoConv1d(input, kernel, stride, dilation);
}

int main() {
  nice(-20);

  Tensor input = CreateTensor(2048);
  Tensor kernel = CreateTensor(3);
  Snake snake = CreateSnake(16);

  kernel.data[0] = 1.f;
  kernel.data[1] = 2.f;
  kernel.data[2] = 3.f;

  clock_t start = clock();

  s32 stride = 3;
  s32 dilation = 3;

  {
    printf("Length: %d\n", input.len);

    DoConv1d(&input, &kernel, stride, dilation);

    DoSnake(&input, &snake);
    DoConv1d(&input, &kernel, stride, dilation);
    DoDilatedUnit(&input, &kernel, stride, dilation);
    DoDilatedUnit(&input, &kernel, stride, dilation);

    DoSnake(&input, &snake);
    DoConv1d(&input, &kernel, stride, dilation);
    DoDilatedUnit(&input, &kernel, stride, dilation);
    DoDilatedUnit(&input, &kernel, stride, dilation);
    DoDilatedUnit(&input, &kernel, stride, dilation);

    DoSnake(&input, &snake);
    DoConv1d(&input, &kernel, stride, dilation);
    DoDilatedUnit(&input, &kernel, stride, dilation);
    DoDilatedUnit(&input, &kernel, stride, dilation);
    DoDilatedUnit(&input, &kernel, stride, dilation);

    DoSnake(&input, &snake);
    DoConv1d(&input, &kernel, stride, dilation);
    DoDilatedUnit(&input, &kernel, stride, dilation);
    DoDilatedUnit(&input, &kernel, stride, dilation);
    DoDilatedUnit(&input, &kernel, stride, dilation);

    DoSnake(&input, &snake);
    DoConv1d(&input, &kernel, stride, dilation);
  }

  clock_t end = clock();
  f64 time = ((f64)(end - start) / CLOCKS_PER_SEC);

  printf("Took: %f\n", time);
}

