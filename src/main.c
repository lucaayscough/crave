#include "crave.h"
#include "models/v1.h"

int main(int argc, char* argv[]) {
  arena_t arena = {};
  arena_init(&arena, 5 * MB);

  load_weights(&arena);

  clock_t start = clock();

  tensor_t* z = tensor_create(&arena, U32_TPL(1, 4, 1), 4096);

  for (int i = 0; i < 16; ++i) {
    tensor_init(z, U32_TPL(1, 4, 1));

    z->data[0] = 0.1f;
    z->data[1] = 0.2f;
    z->data[2] = 0.3f;
    z->data[3] = 0.4f;

    decode(z);
  }

  print_runtime_ms(start);
  arena_free(&arena);

  return 0;
}
