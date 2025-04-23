#ifndef ARENA_H
#define ARENA_H

#define KB 1000
#define MB 1000000
#define GB 1000000000

typedef struct {
  size_t size;
  uint64_t index;
  void* data;
} arena_t;

// TODO(luca): Add allocation boundaries to validate memory writes.
// TODO(luca): Allocate memory normally to validate memory writes.
// TODO(luca): Use base pointer.
// TODO(luca): Make non-global.
// TODO(luca): Add option to commit memory when allocating.

static arena_t g_arena;

void arena_init(size_t size) {
  void* data = calloc(size, 1);

  if (data) {
    g_arena.size = size;
    g_arena.index = 0;
    g_arena.data = data;
  }
}

void* arena_alloc(size_t size) {
  size_t alligned_size = size + (sizeof(uint64_t) - size % sizeof(uint64_t));
  assert(alligned_size % sizeof(uint64_t) == 0);

  // TODO(luca): Add boundary bytes for safety.
  if (g_arena.index + alligned_size > g_arena.size) {
    LOG_AND_ASSERT("Not enough memory.");
    return NULL;
  }

  char* data = ((char*)(g_arena.data))[g_arena.index];
  g_arena.index += alligned_size;

  return data;
}

void arena_free() {
  free(g_arena.data); 
}

#endif // ARENA_H
