#ifndef FILE_H
#define FILE_H

#include <dirent.h>

// TODO(luca): Tidy this up.
#define MAX_FILE_PATH_COUNT 1024

typedef struct {
  char** paths;
  size_t capacity;
  size_t count;
} file_list_t;

// TODO(luca): Add get_num_files_in_dir
// TODO(luca): Add get_dir_exists
// TODO(luca): Pass our own memory to file_list_create_from_dir

file_list_t* file_list_create_from_dir(arena_t* arena, char* dir_path) {
  size_t dir_path_len = strlen(dir_path);
  DIR* dir = opendir(dir_path);
  if (dir == NULL) {
    return NULL;
  }

  file_list_t* list = (file_list_t*)arena_alloc(arena, sizeof(file_list_t));
  if (list == NULL) {
    closedir(dir);
    return NULL;
  }

  list->paths = (char**)arena_alloc(arena, MAX_FILE_PATH_COUNT * sizeof(char*));
  if (list->paths == NULL) {
    closedir(dir);
    return NULL;
  }

  list->capacity = MAX_FILE_PATH_COUNT;
  list->count = 0;

  struct dirent* entry;
  while ((entry = readdir(dir))) {
    if (strcmp(".", entry->d_name) == 0 || strcmp("..", entry->d_name) == 0) {
      continue;
    }
    
    size_t entry_len = strlen(entry->d_name);
    size_t len = dir_path_len + entry_len + 1;

    list->paths[list->count] = (char*)arena_alloc(arena, len);
    if (list->paths[list->count] == NULL) {
      closedir(dir);
      return NULL;
    }

    size_t cpy_count = strlcat(list->paths[list->count], dir_path, len);
    if (cpy_count != dir_path_len) {
      closedir(dir);
      return NULL;
    }

    cpy_count = strlcat(list->paths[list->count], entry->d_name, len); 
    if (cpy_count != len - 1) {
      closedir(dir);
      return NULL;
    }

    ++list->count;
    
    if (list->count == MAX_FILE_PATH_COUNT) {
      LOG_W("Returning early because the file count exceeded MAX_FILE_PATH_COUNT.");
      break;
    }
  }

  closedir(dir);
  return list; 
}

off_t get_file_size(FILE* file) {
  int fd = fileno(file);
  assert(fd);
  
  struct stat st;
  fstat(fd, &st); 

  return st.st_size;
}

#endif // FILE_H
