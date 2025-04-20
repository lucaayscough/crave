#ifndef DEBUG_H
#define DEBUG_H

#define breakpoint __builtin_trap()
#define BREAK_IF(cond) do { if (cond) { breakpoint; }; } while (0)
#define VOID

#define CHECK(e, r, ...) \
  do { \
    assert(e); \
    if (!(e)) { \
      LOG_E(__VA_ARGS__); \
      return r; \
    } \
  } while (0)

#define CHECK_GOTO(e, label, ...) \
  do { \
    assert(e); \
    if (!(e)) { \
      LOG_E(__VA_ARGS__); \
      goto label; \
    } \
  } while (0)

#define LOG_AND_ASSERT(...) do { LOG_E(__VA_ARGS__); assert(0); } while (0)

#ifdef INTERNAL
 #define DO_INTERNAL(func) do { func } while (0)
#else
 #define DO_INTERNAL(func)
#endif

#endif // DEBUG_H
