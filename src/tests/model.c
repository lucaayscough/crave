#define MODEL_TEST
//#define CRV_TENSOR_CONV1D_PRINT_IM2COL_SIZE
//#define CRV_TENSOR_CONV1D_PRINT_ELAPSED_TIME
//#define CRV_TENSOR_CONV_TRANSPOSE1D_PRINT_ELAPSED_TIME
#include "../model.h"

int main(int argc, char* argv[]) {
  printf("\n");
  printf("Testing Model\n");

  void* memory = malloc(150 * CRV_MB);
  if (memory) {
    size_t size = 0;
    if (model_load_file_into_memory(memory, argv[1], &size) == MODEL_LOAD_SUCCESS) {
      model_t model;

      char* file_data = (char*)memory;
      char* it = file_data + size;

      if (model_load(&it, file_data, &model) == MODEL_LOAD_SUCCESS) {
        header_print(&model.header);

        // NOTE(luca): This is a hack, I just know it's here.
        tensor_list_t* list = (tensor_list_t*)(file_data + size);

        tensor_t* z = crv_tensor_find_in_list(list, "z");
        assert(z != NULL);
        tensor_t* y = crv_tensor_find_in_list(list, "y");
        assert(y != NULL);

        crv_tensor_print_shape(z);
        tensor_t* input = crv_tensor_create(&it, CRV_TPL(16 * 2048), CRV_TENSOR_AUTO_CAP, CRV_SWAP);

        size_t alloc_size = crv_alloc_get_size(it, (char*)memory);
        printf("Total alloc size: %zu\n", alloc_size);

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
