#ifndef MEMORY_H
#define MEMORY_H

#define KB 1024
#define MB (KB * KB)
#define GB (MB * MB)

typedef struct {
  size_t size;
  uint64_t index;
  void* data;
} memory_t;

static memory_t g_memory;

void memory_init(size_t size) {
  void* data = calloc(size, 1);

  if (data) {
    g_memory.size = size;
    g_memory.index = 0;
    g_memory.data = data;
  }
}

void* memory_alloc(size_t size) {
  size_t alligned_size = size + (sizeof(uint64_t) - size % sizeof(uint64_t));
  assert(alligned_size % sizeof(uint64_t) == 0);

  // TODO(luca): Add boundary bytes for safety.
  if (g_memory.index + alligned_size > g_memory.size) {
    LOG_AND_ASSERT("Not enough memory.");
    return NULL;
  }

  void* data = &g_memory.data[g_memory.index];
  g_memory.index += alligned_size;

  return data;
}

#endif // MEMORY_H
