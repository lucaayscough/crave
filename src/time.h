#ifndef TIME_H
#define TIME_H

void print_runtime_ms(clock_t start) {
  clock_t end = clock();
  double runtime = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
  printf("Runtime: %.4fms\n", runtime);
}

#endif // TIME_H
