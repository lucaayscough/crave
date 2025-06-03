#define CRV_IMPLEMENTATION
#define IM2COL
#include "model.h"

int main(int argc, char* argv[]) {
  arena_t arena = {};
  crv_arena_init(&arena, 150 * CRV_MB);

  {
    printf("Testing V1\n");

    FILE* file = fopen("/Users/lucaayscough/Desktop/weights_v1.bin", "rb"); 
    if (file) {
      fseek(file, 0, SEEK_END);
      size_t size = ftell(file);
      rewind(file);
      if (size) {
        void* memory = malloc(size);
        if (memory) {
          if (fread(memory, 1, size, file) == size) {
            header_t header = model_get_header_from_memory(memory); 
            
            printf("Model header:\n");
            printf("Size:         %llu\n", header.size);
            printf("Config:       %llu\n", header.config);
            printf("Block size:   %u\n",   header.block_size);
            printf("Num latents:  %u\n",   header.num_latents);
            printf("Sample rate:  %u\n",   header.sample_rate);
            printf("Num tensors:  %u\n",   header.num_tensors);
            printf("\n");

            tensor_list_t* list = crv_tensor_load_from_memory(&arena, (char*)memory + header.size, header.num_tensors);

            if (list) {
              for (int i = 0; i < header.num_tensors; ++i) {
                printf("Found: %s\n", list->tensors[i]->name);
              }

              v1_model_t model = {};
              
              if (v1_load(&arena, &model, list) == MODEL_LOAD_SUCCESS) {
                printf("Success!\n");

                tensor_t* z = crv_tensor_find_in_list(list, "z");
                assert(z != NULL);
                tensor_t* y = crv_tensor_find_in_list(list, "y");
                assert(y != NULL);

                tensor_t* input = crv_tensor_create(&arena, CRV_TPL(1), 16 * 2048);

                clock_t start = clock();

                crv_tensor_copy(input, z);
                v1_decode(input, &model);

                crv_tensor_print_error_stats(input, y);
                crv_print_runtime_ms(start);

                printf("\n");

              } else {
                fprintf(stderr, "Failed to load model.\n");
              }
            } else {
              fprintf(stderr, "Error loading tensor list.\n");
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

    printf("\n\n");
  }

  //{
  //  printf("Testing V2\n");
  //  v2_model_t v2_weights = {};
  //  tensor_list_t* list = NULL;

  //  FILE* file = fopen("weights.bin", "rb"); 
  //  if (file) {
  //    fseek(file, 0, SEEK_END);
  //    size_t size = ftell(file);
  //    rewind(file);
  //    if (size) {
  //      void* memory = malloc(size);
  //      if (memory) {
  //        if (fread(memory, 1, size, file) == size) {
  //          list = crv_tensor_load_from_memory(&arena, memory, size);
  //          if (list) {
  //            v2_load_weights(&arena, &v2_weights, list);
  //          } else {
  //            fprintf(stderr, "Failed to load tensor list.\n");
  //          }
  //        } else {
  //          fprintf(stderr, "Failed to read entire file.\n");
  //        }
  //      } else {
  //        fprintf(stderr, "Couldn't allocate %zu bytes.\n", size);
  //      }
  //      free(memory);
  //    } else {
  //      fprintf(stderr, "Weights file is empty.\n");
  //    }
  //  }

  //  fclose(file); 

  //  tensor_t* z = crv_tensor_find_in_list(list, "z");
  //  assert(z != NULL);
  //  tensor_t* y = crv_tensor_find_in_list(list, "y");
  //  assert(y != NULL);

  //  crv_tensor_print_shape(z);

  //  tensor_t* input = crv_tensor_create(&arena, CRV_TPL(1), 16 * 2048);
  //  uint32_t iters = 1;

  //  clock_t start = clock();

  //  for (int i = 0; i < iters; ++i) {
  //    crv_tensor_copy(input, z);
  //    v2_decode(input, &v2_weights);
  //  }

  //  crv_print_avg_runtime_ms(start, iters);
  //  crv_tensor_print_error_stats(input, y);
  //}

  crv_arena_free(&arena);
  return 0;
}
