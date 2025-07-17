#include "../model.h"

int main(int argc, char* argv[]) {
  printf("\n");
  printf("Model\n");

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
        tensor_t* input = crv_tensor_create(&it, CRV_TPL(1, 32, 1), 16 * 2048, CRV_SWAP);
        model_decode(input, &model);
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
