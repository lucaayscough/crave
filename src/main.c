#define CRV_IMPLEMENTATION
#define CRV_IM2COL
#include "crave.h"
#include "models/v1.h"
#include "models/v2.h"

int main(int argc, char* argv[]) {
  arena_t arena = {};
  crv_arena_init(&arena, 150 * CRV_MB);

  {
    printf("Testing V1\n");
    v1_model_weights_t v1_weights = {};
    v1_load_weights(&arena, &v1_weights);

    tensor_t* z = crv_tensor_load_from_file(&arena, V1_BIN_PATH"z.bin", 4096);
    assert(z != NULL);
    tensor_t* y = crv_tensor_load_from_file(&arena, V1_BIN_PATH"y.bin", CRV_TENSOR_AUTO_CAP);
    assert(y != NULL);

    clock_t start = clock();
    v1_decode(z, &v1_weights);
    crv_tensor_print_error_stats(z, y);
    crv_print_runtime_ms(start);
    printf("\n");
  }

  {
    printf("Testing V2\n");
    v2_model_t v2_weights = {};
    tensor_list_t* list = NULL;

    FILE* file = fopen("weights.bin", "rb"); 
    if (file) {
      fseek(file, 0, SEEK_END);
      size_t size = ftell(file);
      rewind(file);
      if (size) {
        void* memory = malloc(size);
        if (memory) {
          if (fread(memory, 1, size, file) == size) {
            list = crv_tensor_load_from_memory(&arena, memory, size);
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

    tensor_t* z = crv_tensor_find_in_list(list, "z");
    assert(z != NULL);
    tensor_t* y = crv_tensor_find_in_list(list, "y");
    assert(y != NULL);

    crv_tensor_print_shape(z);

    tensor_t* input = crv_tensor_create(&arena, CRV_TPL(1), 16 * 2048);
    uint32_t iters = 1;

    clock_t start = clock();

    for (int i = 0; i < iters; ++i) {
      crv_tensor_copy(input, z);
      v2_decode(input, &v2_weights);
    }

    crv_print_avg_runtime_ms(start, iters);
    crv_tensor_print_error_stats(input, y);
  }

  crv_arena_free(&arena);
  return 0;
}
