#!/usr/bin/env python3

import sys
from typing import List
import struct
import torch
import argparse
import time
import torchaudio

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
  x = torchaudio.load("/Users/lucaayscough/Music/Sound/bjork_hunter_80bpm_stereo.wav")
  x = x[0][0]
  x = x[:2048*2048]
  x = x.reshape(-1, 1, 2048)

  y = torch.zeros(1, 1, 2048)

  model = torch.jit.load("/Users/lucaayscough/Desktop/rave/runs/mrbill-mob-v2-nonoise-cap16/mrbill-mob-v2-nonoise-cap16_4e3d996d79/version_0/checkpoints/mrbill-mob-v2-nonoise-cap16_4e3d996d79.ts")
  model.eval()

  for x_ in x:
    x_ = x_.unsqueeze(0)
    y = torch.cat((y, model.forward(x_)))

  y = y / torch.max(torch.abs(y))
  y = (y * 32767).to(torch.int16)
  y = y.reshape(1, -1).detach().numpy()
  torchaudio.save("audio.wav", y, 44100)

