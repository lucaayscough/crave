#define CRV_IMPLEMENTATION
#include "../model.h"

int main(int argc, char* argv[]) {
  printf("\n");
  printf("Testing Model V1\n");

  void* memory = malloc(5 * CRV_MB);
  if (memory) {
    FILE* file = fopen(argv[1], "rb"); 
    if (file) {
      fseek(file, 0, SEEK_END);
      size_t size = ftell(file);
      rewind(file);
      if (size) {
        if (fread(memory, 1, size, file) == size) {
          header_t header;
          model_get_header_from_memory(&header, memory); 
          
          printf("\n");
          printf("Model header:\n");
          printf(" Size:         %llu\n", header.size);
          printf(" Config:       %llu\n", header.config);
          printf(" Block size:   %u\n",   header.block_size);
          printf(" Num latents:  %u\n",   header.num_latents);
          printf(" Sample rate:  %u\n",   header.sample_rate);
          printf(" Num tensors:  %u\n",   header.num_tensors);
          printf("\n");

          char* read_ptr = (char*)memory + header.size;
          char* write_ptr = (char*)memory + size;

          tensor_list_t* list = crv_tensor_load_from_memory(&write_ptr, read_ptr, header.num_tensors);

          if (list) {
            v1_model_t model = {};
            
            if (v1_load(&write_ptr, &model, list) == MODEL_LOAD_SUCCESS) {
              tensor_t* z = crv_tensor_find_in_list(list, "z");
              assert(z != NULL);
              tensor_t* y = crv_tensor_find_in_list(list, "y");
              assert(y != NULL);

              tensor_t* input = crv_tensor_create_(&write_ptr, CRV_TPL(16 * 2048), CRV_TENSOR_AUTO_CAP, CRV_SWAP);

              clock_t start = clock();

              crv_tensor_copy(input, z);
              v1_decode(input, &model);

              crv_tensor_print_error_stats(input, y);
              printf("\n");
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
        fprintf(stderr, "Weights file is empty.\n");
      }
    } else {
      fprintf(stderr, "Failed to open file.\n");
    }

    fclose(file); 
  }

  return 0;
}
