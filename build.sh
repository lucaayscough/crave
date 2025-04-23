#!/usr/bin/env bash

mkdir -p out

if [ -n "$BUILD_UTILS" ] && [ "$BUILD_UTILS" -eq 1 ]; then
  clang -g -O0 -Wall -Wno-unused-function src/pack.c -o out/pack
fi

if [ -n "$FAST" ] && [ "$FAST" -eq 1 ]; then
  clang -g -O3 -Wall -Wno-unused-function src/main.c -o out/main
else
  clang -g -O0 -D INTERNAL -Wall -Wno-unused-function -Wno-unused-variable src/main.c -o out/main
fi

if [ $? -eq 0 ]; then
  out/main
fi
