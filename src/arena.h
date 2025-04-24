#ifndef ARENA_H
#define ARENA_H

#define KB 1000
#define MB 1000000
#define GB 1000000000

typedef struct {
  size_t size;
  size_t index;
  void* data;
} arena_t;

// TODO(luca): Add allocation boundaries to validate memory writes.
// TODO(luca): Allocate memory normally to validate memory writes.
// TODO(luca): Use base pointer.
// TODO(luca): Add option to commit memory when allocating.
// TODO(luca): While having the arena is nice, it might be nicer to be able to
// simply use our own memory and not have to rely on an allocator. This would
// make the library more flexible for other users that may not want to rely on
// our allocator and instead manage their own memory.

void arena_init(arena_t* arena, size_t size) {
  void* data = calloc(size, 1);

  if (data) {
    arena->size = size;
    arena->index = 0;
    arena->data = data;
  }
}

void* arena_alloc(arena_t* arena, size_t size) {
  size_t alligned_size = size + (sizeof(size_t) - size % sizeof(size_t));
  assert(alligned_size % sizeof(size_t) == 0);

  if (arena->index + alligned_size > arena->size) {
    return NULL;
  }

  void* data = arena->data + arena->index;
  arena->index += alligned_size;

  void* memory = malloc(size);
  return memory;
}

void arena_free(arena_t* arena) {
  free(arena->data); 
}

#endif // ARENA_H
