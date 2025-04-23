#include "crave.h"
#include "models/v1.h"

void run_tests() {
  tensor_t* x = tensor_load_from_file(BIN_PATH"x_test.bin", 1000);
  tensor_t* y = tensor_load_from_file(BIN_PATH"y_test.bin", 0);
  tensor_print_error_stats(x, y);
}

int main(int argc, char* argv[]) {
  arena_init(20 * MB);
 
  load_weights();

  timer_start();

  tensor_t* z = tensor_load_from_file(BIN_PATH"z.bin", 2 * 2048);
  tensor_t* zr = tensor_load_from_file(BIN_PATH"zr.bin", 0);
  decode(z);

  get_runtime_ms();

  tensor_print_error_stats(z, zr);

  arena_free();

  return 0;
}
