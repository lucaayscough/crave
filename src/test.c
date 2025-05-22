#define CRAVE_IMPLEMENTATION
#include "crave.h"

int main() {
  arena_t arena = {};
  arena_init(&arena, 10000);

  tensor_t* x = tensor_create(&arena, U32_TPL(1, 2, 4), 100);
  tensor_t* w = tensor_create(&arena, U32_TPL(2, 2, 2), TENSOR_AUTO_CAP);
  assert(x);
  assert(w); 

  tensor_print_shape(x);
  tensor_print_shape(w);

  tensor_arange(x, 0.f, 1.f);
  tensor_arange(w, 0.f, 1.f);

  tensor_print_data(x);
  tensor_print_data(w);

  tensor_conv_transpose1d(x, w, 1, 1);
  tensor_print_shape(x);
  tensor_print_data(x);
}
