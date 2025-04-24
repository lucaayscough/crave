#include "crave.h"

// TODO(luca): Add output bin.
// TODO(luca): This was written in a hurry.
// TODO(luca): Validate that each binary is a valid tensor.
int main(int argc, char* argv[]) {
  if (argc > 2) {
    fprintf(stderr, "Error: -h for help.\n"); 
    exit(1);
  }

  if (strcmp(argv[1], "-h") == 0) {
    printf("Usage:\n");
    printf(" pack [-h] [path]\n");
    printf("\n");
    printf("Options:\n");
    printf(" -h      Display help message.\n");
    printf("\n");
    printf("Arguments:\n");
    printf(" path    Path to bin directory.\n");
    printf("\n");
    return 0;
  } else {
    arena_t arena = {};
    arena_init(&arena, MB);
    void* memory = malloc(GB);

    file_list_t* list = file_list_create_from_dir(&arena, argv[1]);
    if (list == NULL) {
      fprintf(stderr, "Error creating file list.\n");  
      free(memory);
      arena_free(&arena);
      exit(1);
    }

    FILE* bin = fopen("weights.bin", "wb");
    if (bin == NULL) {
      fprintf(stderr, "Error creating weights bin.\n");  
      fclose(bin);
      free(memory);
      arena_free(&arena);
      exit(1);
    }

    int skipped = 0;
    uint32_t count = list->count;

    fseek(bin, sizeof(uint32_t), 0);

    for (int i = 0; i < list->count; ++i) {
      // TODO(luca): Exit if path is invalid.
      char* path = list->paths[i];
      assert(path != NULL);

      int path_len = strlen(path);
      if (strcmp(path + path_len - 4, ".bin") != 0) {
        ++skipped;
        continue;
      }

      printf("Packing: %s\n", path);

      FILE* file = fopen(path, "rb");
      if (file == NULL) {
        fprintf(stderr, "Failed to open binary file.\n");  
        fclose(bin);
        fclose(file);
        free(memory);
        arena_free(&arena);
        exit(1);
      }

      fseek(file, 0, SEEK_END);
      size_t size = ftell(file);
      fseek(file, 0, SEEK_SET);
      size_t result = fread(memory, 1, size, file);
      if (result != size) {
        fprintf(stderr, "Failed to read data.\n");  
        fclose(bin);
        fclose(file);
        free(memory);
        arena_free(&arena);
        exit(1);
      }

      size_t written = fwrite(memory, 1, size, bin);
      if (written != size) {
        fprintf(stderr, "Failed to write data.\n");  
        fclose(bin);
        fclose(file);
        free(memory);
        arena_free(&arena);
        exit(1);
      }

      fclose(file);
    }

    rewind(bin);
    count -= skipped;
    size_t written = fwrite(&count, sizeof(uint32_t), 1, bin);
    if (written != 1) {
      fprintf(stderr, "Error writing tensor count to file.\n");  
      fclose(bin);
      free(memory);
      arena_free(&arena);
      exit(1);
    }
    
    arena_free(&arena);
    fclose(bin);
    free(memory);
  }

  return 0;
}
