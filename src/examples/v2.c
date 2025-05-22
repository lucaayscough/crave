#define CRAVE_IMPLEMENTATION
#include "../crave.h"
#include "../models/v2.h"

int main(int argc, char* argv[]) {
  arena_t arena = {};
  arena_init(&arena, 500 * MB);

  v2_model_t v2_weights = {};
  v2_load_packed_weights(&arena, &v2_weights);
  tensor_t* z = tensor_load_from_file(&arena, V2_BIN_PATH"z.bin", 16 * 2048);
  assert(z != NULL);

  tensor_print_shape(z);

  tensor_t* input = tensor_create(&arena, U32_TPL(1), 16 * 2048);
  uint32_t iters = 1;

  clock_t start = clock();

  for (int i = 0; i < iters; ++i) {
    tensor_copy(input, z);
    v2_decode(input, &v2_weights);
  }

  print_avg_runtime_ms(start, iters);
  arena_free(&arena);

  return 0;
}
