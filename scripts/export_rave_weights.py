#!/usr/bin/env python3

import torch
import argparse
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

def main():
    parser = argparse.ArgumentParser("Export a RAVE TorchScript file to the CRAVE format.")
    parser.add_argument("-i", "--input", help="Path to TorchScript file.", required=True)
    parser.add_argument("-o", "--output", help="Path to output directory.", required=True)

    args = parser.parse_args()

    model = torch.jit.load(args.input)

    excluded = [
        "learn_target_params",
        "learn_source_params",
        "fake_adain.learn_x",
        "fake_adain.learn_y",
        "fake_adain.mean_x",
        "fake_adain.mean_y",
        "fake_adain.num_update_x",
        "fake_adain.num_update_y",
        "fake_adain.std_x",
        "fake_adain.std_y",
        "fidelity",
        "forward_params",
        "pqmf.h",
        "pqmf.hk",
        "receptive_field",
        "reset_source_params",
        "reset_target_params",
        "decode_params",
        "encode_params",
        "pqmf.forward_conv.cache.pad",
        "pqmf.forward_conv.weight",
        "decoder.noise_module.target_size"
    ]

    params = list(model.named_parameters())
    buffers = list(model.named_buffers())

    for i, (name, param) in enumerate(params):
        if name in excluded or name[:7] == "encoder":
            continue
        tensor_to_file(name, param, args.output)

    for i, (name, buffer) in enumerate(buffers):
        if len(buffer.shape) > 1 and buffer.shape[-1] == 0:
            continue
        if name in excluded or name[:7] == "encoder":
            continue
        if name[-9:] == "cache.pad" or name[-5:] == "cache" or name[-3:] == "pad":
            buffer = buffer[0].unsqueeze(0)
        tensor_to_file(name, buffer, args.output)

if __name__ == "__main__":
    main()
