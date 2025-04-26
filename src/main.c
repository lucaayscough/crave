#define CRAVE_IMPLEMENTATION
#include "crave.h"
#include "models/v1.h"

int main(int argc, char* argv[]) {
  arena_t arena = {};
  arena_init(&arena, 5 * MB);
  v1_model_weights_t weights = {};
  load_weights(&arena, &weights);
  //tensor_t* z = tensor_load_from_file(&arena, BIN_PATH"z.bin", 4096);
  //assert(z != NULL);
  //tensor_t* zr = tensor_load_from_file(&arena, BIN_PATH"zr.bin", TENSOR_AUTO_CAP);
  //assert(zr != NULL);
  //decode(z, &weights);
  //tensor_print_error_stats(z, zr);

  clock_t start = clock();
  tensor_t* z = tensor_create(&arena, U32_TPL(1, 4, 1), 4096);
  for (int i = 0; i < 16; ++i) {
    tensor_init(z, U32_TPL(1, 4, 1));
    z->data[0] = 5.1f;
    z->data[1] = 5.2f;
    z->data[2] = 5.3f;
    z->data[3] = 5.4f;
    decode(z, &weights);
  }
  print_runtime_ms(start);
  arena_free(&arena);
  return 0;
}
