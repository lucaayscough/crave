#define CRAVE_IMPLEMENTATION
#define CRV_IM2COL
#include "../crave.h"
#include "../models/v2.h"

int main(int argc, char* argv[]) {
  arena_t arena = {};
  arena_init(&arena, 500 * MB);

  v2_model_t v2_weights = {};

  FILE* file = fopen("weights.bin", "rb"); 
  if (file) {
    fseek(file, 0, SEEK_END);
    size_t size = ftell(file);
    rewind(file);
    if (size) {
      void* memory = malloc(size);
      if (memory) {
        if (fread(memory, 1, size, file) == size) {
          tensor_list_t* list = tensor_load_from_memory(&arena, memory, size);
          if (list) {
            v2_load_weights(&arena, &v2_weights, list);
          } else {
            fprintf(stderr, "Failed to load tensor list.\n");
          }
        } else {
          fprintf(stderr, "Failed to read entire file.\n");
        }
      } else {
        fprintf(stderr, "Couldn't allocate %zu bytes.\n", size);
      }
      free(memory);
    } else {
      fprintf(stderr, "Weights file is empty.\n");
    }
  }

  fclose(file); 

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
