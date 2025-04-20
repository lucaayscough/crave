#!/usr/bin/env bash

mkdir -p out

if [ -n "$FAST" ] && [ "$FAST" -eq 1 ]; then
  clang -g -O3 -Wall src/main.c -o out/main
else
  clang -g -O0 -D INTERNAL -Wall src/main.c -o out/main
fi

if [ $? -eq 0 ]; then
  out/main
fi

