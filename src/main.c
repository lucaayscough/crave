#define CRAVE_IMPLEMENTATION
#include "crave.h"
#include "models/v1.h"
#include "models/v2.h"

int main(int argc, char* argv[]) {
  arena_t arena = {};
  arena_init(&arena, 500 * MB);

  {
    printf("Testing V1\n");
    v1_model_weights_t v1_weights = {};
    v1_load_weights(&arena, &v1_weights);

    tensor_t* z = tensor_load_from_file(&arena, V1_BIN_PATH"z.bin", 4096);
    assert(z != NULL);
    tensor_t* y = tensor_load_from_file(&arena, V1_BIN_PATH"y.bin", TENSOR_AUTO_CAP);
    assert(y != NULL);

    clock_t start = clock();
    v1_decode(z, &v1_weights);
    tensor_print_error_stats(z, y);
    print_runtime_ms(start);
    printf("\n\n");
  }

  {
    printf("Testing V2\n");
    v2_model_t v2_weights = {};
    v2_load_weights(&arena, &v2_weights);

    tensor_t* z = tensor_load_from_file(&arena, V2_BIN_PATH"z.bin", 16 * 2048);
    assert(z != NULL);
    tensor_t* y = tensor_load_from_file(&arena, V2_BIN_PATH"y.bin", TENSOR_AUTO_CAP);
    assert(y != NULL);

    tensor_print_shape(z);

    tensor_t* input = tensor_create(&arena, U32_TPL(1), 16 * 2048);
    uint32_t iters = 1;

    clock_t start = clock();

    for (int i = 0; i < iters; ++i) {
      tensor_copy(input, z);
      v2_decode(input, &v2_weights);
    }

    print_avg_runtime_ms(start, iters);

    tensor_print_error_stats(input, y);
  }

  arena_free(&arena);
  return 0;
}
