#!/usr/bin/env bash

mkdir -p out

CFLAGS="-g -Wall -O0 -DCRV_INTERNAL -DCRV_IM2COL"
CPPFLAGS="-Wno-deprecated -std=c++20"

clang $CFLAGS src/tests/v1.c -o out/tests_v1 && out/tests_v1 out/bin/v1_test_weights.bin
clang $CFLAGS src/tests/v2.c -o out/tests_v2 && out/tests_v2 out/bin/v2_test_weights.bin
clang $CFLAGS src/pack.c -o out/pack

clang++ $CPPFLAGS $CFLAGS src/tests/v1.c -o out/__cpp_tests_v1
clang++ $CPPFLAGS $CFLAGS src/tests/v2.c -o out/__cpp_tests_v2
clang++ $CPPFLAGS $CFLAGS src/pack.c -o out/pack
