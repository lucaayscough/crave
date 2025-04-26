#define CRAVE_IMPLEMENTATION
#include "crave.h"
#include "models/v1.h"
#include "models/v2.h"

int main(int argc, char* argv[]) {
  arena_t arena = {};
  arena_init(&arena, 500 * MB);

  {
    v1_model_weights_t v1_weights = {};
    v1_load_weights(&arena, &v1_weights);

    tensor_t* z = tensor_load_from_file(&arena, V1_BIN_PATH"z.bin", 4096);
    assert(z != NULL);
    tensor_t* zr = tensor_load_from_file(&arena, V1_BIN_PATH"zr.bin", TENSOR_AUTO_CAP);
    assert(zr != NULL);

    clock_t start = clock();
    v1_decode(z, &v1_weights);
    tensor_print_error_stats(z, zr);
    print_runtime_ms(start);
  }

  {
    v2_model_weights_t v2_weights = {};
    v2_load_weights(&arena, &v2_weights);

    tensor_t* z = tensor_load_from_file(&arena, V2_BIN_PATH"z.bin", 4096);
    assert(z != NULL);
    tensor_t* zr = tensor_load_from_file(&arena, V2_BIN_PATH"zr.bin", TENSOR_AUTO_CAP);
    assert(zr != NULL);

    clock_t start = clock();
    v2_decode(z, &v2_weights);
    tensor_print_error_stats(z, zr);
    print_runtime_ms(start);
  }

  arena_free(&arena);
  return 0;
}


