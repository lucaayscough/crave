#define CRV_IMPLEMENTATION
#include "../model.h"

int main(int argc, char* argv[]) {

  printf("\n");
  printf("Testing Model V2\n");

  void* memory = malloc(150 * CRV_MB);
  if (memory) {
    size_t size = 0;
    if (model_load_file_into_memory(memory, argv[1], &size) == MODEL_LOAD_SUCCESS) {
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
        v2_model_t model = {};

        if (v2_load(&write_ptr, &model, list) == MODEL_LOAD_SUCCESS) {
          tensor_t* z = crv_tensor_find_in_list(list, "z");
          assert(z != NULL);
          tensor_t* y = crv_tensor_find_in_list(list, "y");
          assert(y != NULL);

          crv_tensor_print_shape(z);

          tensor_t* input = crv_tensor_create(&write_ptr, CRV_TPL(16 * 2048), CRV_TENSOR_AUTO_CAP, CRV_SWAP);

          clock_t start = clock();

          crv_tensor_copy(input, z);
          v2_decode(input, &model);

          crv_print_runtime_ms(start);
          crv_tensor_print_error_stats(input, y);

        } else {
          fprintf(stderr, "Failed to load model.\n");
        }
      } else {
        fprintf(stderr, "Error loading tensor list.\n");
      }
    } else {
      fprintf(stderr, "Error reading data from file.\n");
    }
  } else {
    fprintf(stderr, "Error allocating memory.\n");
  }

  free(memory);

  return 0;
}
