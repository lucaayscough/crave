#!/usr/bin/env python3

import sys
from typing import List
import struct
import torch
import argparse
import time

def tensor_to_file(name: str, x: torch.Tensor):
  ndim = x.ndim
  dims_list = list(x.shape)
  x_flat = x.float().flatten()
  x_list: List[float] = x_flat.tolist()
  list_size = len(x_list)

  name_bytes = name.encode('utf-8') + b'\0'
  name_length = len(name_bytes)
  header_buf = struct.pack("I", name_length) + name_bytes

  metadata_buf = struct.pack("I", ndim)

  for dim_size in dims_list:
    metadata_buf += struct.pack("I", dim_size)

  metadata_buf += struct.pack("I", list_size)

  flat_buf = struct.pack(f'{list_size}f', *x_list)

  with open("/Users/lucaayscough/dev/crave/out/bin/" + name + ".bin", "wb") as f:
    f.write(header_buf + metadata_buf + flat_buf)

if __name__ == "__main__":
  x = torch.arange(0, 16).reshape(2, 2, -1)
  y = x.flip(1)

  print(x)
  print(y)

  tensor_to_file("x_test", x)
  tensor_to_file("y_test", y)

