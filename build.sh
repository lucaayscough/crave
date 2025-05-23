#!/usr/bin/env bash

mkdir -p out

COMPILER_FLAGS="-Wall -Wno-deprecated -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 -DCRV_IM2COL -framework Accelerate -fopenmp"

if [ -n "$BUILD_UTILS" ] && [ "$BUILD_UTILS" -eq 1 ]; then
  clang -g -O0 src/pack.c -o out/pack
fi

clang++ -std=c++20 src/main.c $COMPILER_FLAGS -o out/__cpp_test

if [ -n "$FAST" ] && [ "$FAST" -eq 1 ]; then
  clang -g -O3 $COMPILER_FLAGS src/main.c -o out/main
else
  clang -g -O0 -DINTERNAL $COMPILER_FLAGS src/main.c -o out/main
fi

if [ $? -eq 0 ]; then
  out/main
fi

