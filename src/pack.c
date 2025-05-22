#define CRAVE_IMPLEMENTATION
#include "crave.h"
#include <sys/dir.h>
#include <string.h>

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
  }

  arena_t arena = {};
  arena_init(&arena, MB);

  void* memory = malloc(GB);
  if (memory) {
    char* dir_path = argv[1];
    size_t len_dir_path = strlen(dir_path);
    DIR* dir = opendir(dir_path);
    if (dir) {
      FILE* bin = fopen("weights.bin", "wb");
      if (bin) {
        uint32_t written_count = 0;
        if (fwrite(&written_count, sizeof(uint32_t), 1, bin) == 1) {
          struct dirent* entry = readdir(dir);
          while (entry) {
            char* filename = entry->d_name;
            size_t len_filename = strlen(filename);
            if (len_filename > 4) {
              if (strcmp(filename + len_filename - 4, ".bin") == 0) {
                size_t len_filepath = len_dir_path + len_filename;
                char* filepath = alloca(len_filepath);
                memset(filepath, 0, len_filepath);
                strcpy(filepath, dir_path);
                strlcat(filepath, filename, len_filepath + 1);
                FILE* file = fopen(filepath, "rb");
                if (file) {
                  printf("Packing: %s\n", filename);
          
                  fseek(file, 0, SEEK_END);
                  size_t size = ftell(file);
                  fseek(file, 0, SEEK_SET);

                  if (size <= GB) {
                    if (fread(memory, 1, size, file) == size) {
                      if (fwrite(memory, 1, size, bin) == size) {
                        ++written_count;
                      } else {
                        fprintf(stderr, "Could not write entire file: %s\n", filename);
                      }
                    } else {
                      fprintf(stderr, "Could not read entire file: %s\n", filename);
                    }
                  } else {
                    fprintf(stderr, "File size too large: %s\n", filename);
                  }
                } else {
                  fprintf(stderr, "Couldn't open file: %s\n", filename);
                }
                fclose(file);
              }
            }
            entry = readdir(dir);
          }
          rewind(bin);
          if (fwrite(&written_count, sizeof(uint32_t), 1, bin) == 1) {
            printf("Written file count: %u.\n", written_count);
          } else {
            fprintf(stderr, "Error writing file count to file.\n");  
          }
        } else {
          fprintf(stderr, "Couldn't write to weights.bin file.\n");
        }
      } else {
        fprintf(stderr, "Couldn't create weights.bin file.\n");
      }
      fclose(bin);
    } else {
      fprintf(stderr, "Couldn't open directory.\n");
    }
    closedir(dir);
  } else {
    fprintf(stderr, "Couldn't allocate memory.\n");
  }
  free(memory);
  arena_free(&arena);

  return 0;
}
