#ifndef TIME_H
#define TIME_H

static clock_t start;

void timer_start() {
  start = clock();
}

void get_runtime_ms() {
  clock_t end = clock();
  double runtime = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
  printf("Runtime: %.4fms\n", runtime);
}

#endif // TIME_H
