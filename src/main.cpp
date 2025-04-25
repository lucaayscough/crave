#include "crave.h"
#include "models/v1.h"

int main(int argc, char* argv[]) {
  arena_t arena = {};
  arena_init(&arena, 5 * MB);

  v1_model_weights_t weights = {};
  tensor_list_t* list = tensor_load_from_blob(&arena, "weights.bin");
  load_weights(&arena, &weights, list);

  clock_t start = clock();

  tensor_t* z = tensor_create(&arena, U32_TPL(1, 4, 1), 4096);

  for (int i = 0; i < 16; ++i) {
    tensor_init(z, U32_TPL(1, 4, 1));

    z->data[0] = 0.1f;
    z->data[1] = 0.2f;
    z->data[2] = 0.3f;
    z->data[3] = 0.4f;

    decode(z, &weights);
  }

  print_runtime_ms(start);
  arena_free(&arena);

  return 0;
}
