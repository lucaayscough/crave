#!/usr/bin/env python3

import torch
import time

#x = torch.randn(1, 1536, 1)
#w = torch.randn(1536, 768, 4)
#y = torch.conv_transpose1d(x, w, stride=2, dilation=1)
#
#x = torch.randn(1, 768, 2)
#w = torch.randn(768, 384, 8)
#y = torch.conv_transpose1d(x, w, stride=4, dilation=1)
#
#x = torch.randn(1, 384, 8)
#w = torch.randn(384, 192, 8)
#y = torch.conv_transpose1d(x, w, stride=4, dilation=1)
#
#x = torch.randn(1, 192, 32)
#w = torch.randn(192, 96, 8)
#y = torch.conv_transpose1d(x, w, stride=4, dilation=1)


import torch
import torch.nn.functional as F
import struct
from pathlib import Path

def tensor_to_file(name: str, x: torch.Tensor, dir: str):
    x = x.contiguous()
    ndim = x.ndim
    dims_list = list(x.shape)
    x_flat = x.flatten().float()
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

    output_path = Path(dir) / (name + ".bin")
    with open(output_path, "wb") as f:
        f.write(header_buf + metadata_buf + flat_buf)

def simulate_conv_transpose1d(x, w, stride=2, dilation=1):
    batch_size, in_channels, length = x.shape
    kernel_size = w.shape[2]

    upsampled_length = (length - 1) * stride + 1
    upsampled = torch.zeros(batch_size, in_channels, upsampled_length, device=x.device, dtype=x.dtype)
    upsampled[:, :, ::stride] = x
    tensor_to_file("upsampled", upsampled, "out/bin/conv_transpose1d")

    permutated_weights = w.permute(1, 0, 2)
    tensor_to_file("permutated", permutated_weights, "out/bin/conv_transpose1d")

    flipped_weights = permutated_weights.flip(2)
    tensor_to_file("flipped", flipped_weights, "out/bin/conv_transpose1d")

    padding = dilation * (kernel_size - 1)

    y_sim = F.conv1d(upsampled, flipped_weights, padding=padding, dilation=dilation)
    tensor_to_file("y", y_sim, "out/bin/conv_transpose1d")

    return y_sim

x = torch.randn(1, 1536, 1)
w = torch.randn(1536, 768, 4)

tensor_to_file("x", x, "out/bin/conv_transpose1d")
tensor_to_file("w", w, "out/bin/conv_transpose1d")

y_true = F.conv_transpose1d(x, w, stride=2, dilation=1)
y_simulated = simulate_conv_transpose1d(x, w, stride=2, dilation=1)

print("Are the results close?", torch.allclose(y_true, y_simulated, atol=1e-5))
print("Difference:", (y_true - y_simulated).abs().max())
