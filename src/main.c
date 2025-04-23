#include "crave.h"
#include "models/v1.h"

int main(int argc, char* argv[]) {
  arena_init(20 * MB);
  load_weights();
  timer_start();
  tensor_t* z = tensor_create(U32_TPL(1, 4, 1), 2 * 2048);
  decode(z);
  get_runtime_ms();
  arena_free();
  return 0;
}
