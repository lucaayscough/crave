#!/usr/bin/env bash

mkdir -p out

COMPILER_FLAGS="-Wall -Wno-unused-function -Wno-unused-variable"

if [ -n "$BUILD_UTILS" ] && [ "$BUILD_UTILS" -eq 1 ]; then
  clang -g -O0 src/pack.c -o out/pack
fi

if [ -n "$FAST" ] && [ "$FAST" -eq 1 ]; then
  clang++ -g -O3 $COMPILER_FLAGS src/main.c++ -o out/main
else
  clang++ -g -O0 -D INTERNAL $COMPILER_FLAGS src/main.cpp -o out/main
fi

if [ $? -eq 0 ]; then
  out/main
fi
