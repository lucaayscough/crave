#ifndef FILE_H
#define FILE_H

#define MAX_FILE_PATH_COUNT 1024

typedef struct {
  char** paths;
  size_t capacity;
  size_t count;
} file_list_t;

file_list_t* file_list_create_from_dir(char* dir_path) {
  size_t dir_path_len = strlen(dir_path);
  DIR* dir = opendir(dir_path);
  CHECK_GOTO(dir, error, "Couldn't allocate memory.");

  file_list_t* list = arena_alloc(sizeof(file_list_t));
  CHECK_GOTO(list, error, "Couldn't allocate memory.");

  list->paths = arena_alloc(MAX_FILE_PATH_COUNT * sizeof(char*));
  CHECK_GOTO(list->paths, error, "Couldn't allocate memory.");

  list->capacity = MAX_FILE_PATH_COUNT;
  list->count = 0;

  struct dirent* entry;
  while ((entry = readdir(dir))) {
    if (strcmp(".", entry->d_name) == 0 || strcmp("..", entry->d_name) == 0) {
      continue;
    }
    
    size_t entry_len = strlen(entry->d_name);
    size_t len = dir_path_len + entry_len + 1;

    list->paths[list->count] = arena_alloc(len);
    CHECK_GOTO(list->paths[list->count], error, "Couldn't allocate memory.");

    size_t cpy_count = strlcat(list->paths[list->count], dir_path, len);
    CHECK_GOTO(cpy_count == dir_path_len, error, "Failed to copy string.");

    cpy_count = strlcat(list->paths[list->count], entry->d_name, len); 
    CHECK_GOTO(cpy_count == len - 1, error, "Failed to copy string.");

    ++list->count;
    
    if (list->count == MAX_FILE_PATH_COUNT) {
      LOG_W("Returning early because the file count exceeded MAX_FILE_PATH_COUNT.");
      break;
    }
  }

  closedir(dir);

  return list; 
   
error:
  // TODO(luca): Add a way to deallocate the memory.
  closedir(dir);
  return NULL;
}

off_t get_file_size(FILE* file) {
  int fd = fileno(file);
  assert(fd);
  
  struct stat st;
  fstat(fd, &st); 

  return st.st_size;
}

#endif // FILE_H
