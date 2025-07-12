#define CRV_IMPLEMENTATION
#include "../model.h"

int main(int argc, char* argv[]) {
  printf("\n");
  printf("Testing Model V1\n");

  void* memory = malloc(150 * CRV_MB);
  if (memory) {
    size_t size = 0;
    if (model_load_file_into_memory(memory, argv[1], &size) == MODEL_LOAD_SUCCESS) {
      model_t model;

      char* file_data = (char*)memory;
      char* it = file_data + size;

      if (model_load(&it, file_data, &model) == MODEL_LOAD_SUCCESS) {

        printf("\n");
        printf("Model header:\n");
        printf(" Size:         %llu\n", model.header.size);
        printf(" Config:       %llu\n", model.header.config);
        printf(" Block size:   %u\n",   model.header.block_size);
        printf(" Num latents:  %u\n",   model.header.num_latents);
        printf(" Sample rate:  %u\n",   model.header.sample_rate);
        printf(" Num tensors:  %u\n",   model.header.num_tensors);
        printf("\n");

        // NOTE(luca): This is a hack, I just know it's here.
        tensor_list_t* list = (tensor_list_t*)(file_data + size);

        tensor_t* z = crv_tensor_find_in_list(list, "z");
        assert(z != NULL);
        tensor_t* y = crv_tensor_find_in_list(list, "y");
        assert(y != NULL);

        crv_tensor_print_shape(z);
        tensor_t* input = crv_tensor_create(&it, CRV_TPL(16 * 2048), CRV_TENSOR_AUTO_CAP, CRV_SWAP);

        clock_t start = clock();

        crv_tensor_copy(input, z);
        model_decode(input, &model);

        crv_tensor_print_error_stats(input, y);
        printf("\n");
        crv_print_runtime_ms(start);
        printf("\n");
        
      } else {
        fprintf(stderr, "Error loading model.\n");
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
