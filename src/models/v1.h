#ifndef V1_H
#define V1_H

#define BIN_PATH "/Users/lucaayscough/dev/crave/out/bin/"

void pre_process_latent(tensor_t* z) {
  tensor_t* noise = tensor_load_from_file(BIN_PATH"pre_process_latent_noise.bin", 0);
  tensor_t* latent_pca = tensor_load_from_file(BIN_PATH"pre_process_latent_latent_pca.bin", 0);
  tensor_t* latent_mean = tensor_load_from_file(BIN_PATH"pre_process_latent_latent_mean.bin", 0);
  
  tensor_transpose(latent_pca, 0, 1);
  tensor_unsqueeze(latent_pca, latent_pca->rank);
  tensor_cat(z, noise, 1);
  tensor_conv1d(z, latent_pca, 1, 1);

  tensor_unsqueeze(latent_mean, 0);
  tensor_unsqueeze(latent_mean, latent_mean->rank);
  tensor_tadd(z, latent_mean);
}

void post_process_latent(tensor_t* z) {
  tensor_reshape(z, TUPLE(1, 16, 128)); 
  tensor_t* mask = tensor_create(TUPLE(1, 16, 128), 16 * 128);
  tensor_fill(mask, 1.f);

  size_t channels = mask->dims[1];
  size_t len = mask->dims[2];
  
  for (size_t i = 1; i < channels; i += 2) {
    for (size_t j = 0; j < len; j += 2) {
      size_t idx = i * len + j; 
      mask->data[idx] = -1; 
    }
  }

  tensor_tmul(z, mask);
  tensor_pad(z, 32);

  tensor_t* pqmf_inverse_conv_weight = tensor_load_from_file(BIN_PATH"pqmf.inverse_conv.weight.bin", 0);
  tensor_conv1d(z, pqmf_inverse_conv_weight, 1, 1);
  tensor_mul(z, 16);
  tensor_flip(z, 1);
  tensor_permute(z, TUPLE(0, 2, 1));
  tensor_reshape(z, TUPLE(1, 128, 1, 16));
  tensor_permute(z, TUPLE(0, 2, 1, 3));
  tensor_reshape(z, TUPLE(1, 1, 2048));
}

void decode(tensor_t* z) {
  tensor_t* skip = tensor_create(TUPLE(1), 1024 * 1024);
  pre_process_latent(z);

  // (0)
  tensor_t* net_0_weight = tensor_load_from_file(BIN_PATH"decoder.net.0.weight.bin", 0);
  tensor_pad(z, 2);
  tensor_conv1d(z, net_0_weight, 1, 1);

  // (1)
  tensor_t* net_1_alpha = tensor_load_from_file(BIN_PATH"decoder.net.1.alpha.bin", 0);
  tensor_snake(z, net_1_alpha);

  // (2)
  tensor_t* net_2_weight = tensor_load_from_file(BIN_PATH"decoder.net.2.weight.bin", 0);
  tensor_pad(z, 1);
  tensor_conv_transpose1d(z, net_2_weight, 2, 1);
  tensor_trunc(z, 2, 2);

  // (3)
  tensor_copy(skip, z);

  {
    // (0)
    tensor_leaky_relu(z, 0.2f);

    // (1)
    tensor_t* decoder_net_3_aligned_branches_0_net_1_weight = tensor_load_from_file(BIN_PATH"decoder.net.3.aligned.branches.0.net.1.weight.bin", 0);
    tensor_pad(z, 2);
    tensor_conv1d(z, decoder_net_3_aligned_branches_0_net_1_weight, 1, 1);

    // (2)
    tensor_leaky_relu(z, 0.2f);

    // (3)
    tensor_t* decoder_net_3_aligned_branches_0_net_3_weight = tensor_load_from_file(BIN_PATH"decoder.net.3.aligned.branches.0.net.3.weight.bin", 0);
    tensor_conv1d(z, decoder_net_3_aligned_branches_0_net_3_weight, 1, 1);
  }

  tensor_tadd(z, skip);

  // (4)
  tensor_copy(skip, z);

  {
    // (0)
    tensor_leaky_relu(z, 0.2f);

    // (1)
    tensor_t* decoder_net_4_aligned_branches_0_net_1_weight = tensor_load_from_file(BIN_PATH"decoder.net.4.aligned.branches.0.net.1.weight.bin", 0);
    tensor_pad(z, 6);
    tensor_conv1d(z, decoder_net_4_aligned_branches_0_net_1_weight, 1, 3);

    // (2)
    tensor_leaky_relu(z, 0.2f);

    // (3)
    tensor_t* decoder_net_4_aligned_branches_0_net_3_weight = tensor_load_from_file(BIN_PATH"decoder.net.4.aligned.branches.0.net.3.weight.bin", 0);
    tensor_conv1d(z, decoder_net_4_aligned_branches_0_net_3_weight, 1, 1);
  }

  tensor_tadd(z, skip);

  // (5)
  tensor_t* net_5_alpha = tensor_load_from_file(BIN_PATH"decoder.net.5.alpha.bin", 0);
  tensor_snake(z, net_5_alpha);

  // (6)
  tensor_t* net_6_weight = tensor_load_from_file(BIN_PATH"decoder.net.6.weight.bin", 0);
  tensor_pad(z, 1);
  tensor_conv_transpose1d(z, net_6_weight, 4, 1);
  tensor_trunc(z, 4, 4);

  // (7)
  tensor_copy(skip, z);

  {
    // (0)
    tensor_leaky_relu(z, 0.2f);

    // (1)
    tensor_t* decoder_net_7_aligned_branches_0_net_1_weight = tensor_load_from_file(BIN_PATH"decoder.net.7.aligned.branches.0.net.1.weight.bin", 0);
    tensor_pad(z, 2);
    tensor_conv1d(z, decoder_net_7_aligned_branches_0_net_1_weight, 1, 1);

    // (2)
    tensor_leaky_relu(z, 0.2f);

    // (3)
    tensor_t* decoder_net_7_aligned_branches_0_net_3_weight = tensor_load_from_file(BIN_PATH"decoder.net.7.aligned.branches.0.net.3.weight.bin", 0);
    tensor_conv1d(z, decoder_net_7_aligned_branches_0_net_3_weight, 1, 1);
  }

  tensor_tadd(z, skip);
  // (8)
  tensor_copy(skip, z);

  {
    // (0)
    tensor_leaky_relu(z, 0.2f);

    // (1)
    tensor_t* decoder_net_8_aligned_branches_0_net_1_weight = tensor_load_from_file(BIN_PATH"decoder.net.8.aligned.branches.0.net.1.weight.bin", 0);
    tensor_pad(z, 6);
    tensor_conv1d(z, decoder_net_8_aligned_branches_0_net_1_weight, 1, 3);

    // (2)
    tensor_leaky_relu(z, 0.2f);

    // (3)
    tensor_t* decoder_net_8_aligned_branches_0_net_3_weight = tensor_load_from_file(BIN_PATH"decoder.net.8.aligned.branches.0.net.3.weight.bin", 0);
    tensor_conv1d(z, decoder_net_8_aligned_branches_0_net_3_weight, 1, 1);
  }

  tensor_tadd(z, skip);

  // (9)
  tensor_copy(skip, z);

  {
    // (0)
    tensor_leaky_relu(z, 0.2f);

    // (1)
    tensor_t* decoder_net_9_aligned_branches_0_net_1_weight = tensor_load_from_file(BIN_PATH"decoder.net.9.aligned.branches.0.net.1.weight.bin", 0);
    tensor_pad(z, 18);
    tensor_conv1d(z, decoder_net_9_aligned_branches_0_net_1_weight, 1, 9);

    // (2)
    tensor_leaky_relu(z, 0.2f);

    // (3)
    tensor_t* decoder_net_9_aligned_branches_0_net_3_weight = tensor_load_from_file(BIN_PATH"decoder.net.9.aligned.branches.0.net.3.weight.bin", 0);
    tensor_conv1d(z, decoder_net_9_aligned_branches_0_net_3_weight, 1, 1);
  }

  tensor_tadd(z, skip);

  // (10)
  tensor_t* net_10_alpha = tensor_load_from_file(BIN_PATH"decoder.net.10.alpha.bin", 0);
  tensor_snake(z, net_10_alpha);

  // (11)
  tensor_t* net_11_weight = tensor_load_from_file(BIN_PATH"decoder.net.11.weight.bin", 0);
  tensor_pad(z, 1);
  tensor_conv_transpose1d(z, net_11_weight, 4, 1);
  tensor_trunc(z, 4, 4);

  // (12)
  tensor_copy(skip, z);

  {
    // (0)
    tensor_leaky_relu(z, 0.2f);

    // (1)
    tensor_t* decoder_net_12_aligned_branches_0_net_1_weight = tensor_load_from_file(BIN_PATH"decoder.net.12.aligned.branches.0.net.1.weight.bin", 0);
    tensor_pad(z, 2);
    tensor_conv1d(z, decoder_net_12_aligned_branches_0_net_1_weight, 1, 1);

    // (2)
    tensor_leaky_relu(z, 0.2f);

    // (3)
    tensor_t* decoder_net_12_aligned_branches_0_net_3_weight = tensor_load_from_file(BIN_PATH"decoder.net.12.aligned.branches.0.net.3.weight.bin", 0);
    tensor_conv1d(z, decoder_net_12_aligned_branches_0_net_3_weight, 1, 1);
  }

  tensor_tadd(z, skip);

  // (13)
  tensor_copy(skip, z);

  {
    // (0)
    tensor_leaky_relu(z, 0.2f);

    // (1)
    tensor_t* decoder_net_13_aligned_branches_0_net_1_weight = tensor_load_from_file(BIN_PATH"decoder.net.13.aligned.branches.0.net.1.weight.bin", 0);
    tensor_pad(z, 6);
    tensor_conv1d(z, decoder_net_13_aligned_branches_0_net_1_weight, 1, 3);

    // (2)
    tensor_leaky_relu(z, 0.2f);

    // (3)
    tensor_t* decoder_net_13_aligned_branches_0_net_3_weight = tensor_load_from_file(BIN_PATH"decoder.net.13.aligned.branches.0.net.3.weight.bin", 0);
    tensor_conv1d(z, decoder_net_13_aligned_branches_0_net_3_weight, 1, 1);
  }
   
  tensor_tadd(z, skip);

  // (14)
  tensor_copy(skip, z);

  {
    // (0)
    tensor_leaky_relu(z, 0.2f);

    // (1)
    tensor_t* decoder_net_14_aligned_branches_0_net_1_weight = tensor_load_from_file(BIN_PATH"decoder.net.14.aligned.branches.0.net.1.weight.bin", 0);
    tensor_pad(z, 18);
    tensor_conv1d(z, decoder_net_14_aligned_branches_0_net_1_weight, 1, 9);

    // (2)
    tensor_leaky_relu(z, 0.2f);

    // (3)
    tensor_t* decoder_net_14_aligned_branches_0_net_3_weight = tensor_load_from_file(BIN_PATH"decoder.net.14.aligned.branches.0.net.3.weight.bin", 0);
    tensor_conv1d(z, decoder_net_14_aligned_branches_0_net_3_weight, 1, 1);
  }
   
  tensor_tadd(z, skip);

  // (15)
  tensor_t* net_15_alpha = tensor_load_from_file(BIN_PATH"decoder.net.15.alpha.bin", 0);
  tensor_snake(z, net_15_alpha);

  // (16)
  tensor_t* net_16_weight = tensor_load_from_file(BIN_PATH"decoder.net.16.weight.bin", 0);
  tensor_pad(z, 1);
  tensor_conv_transpose1d(z, net_16_weight, 4, 1);
  tensor_trunc(z, 4, 4);

  // (17)
  tensor_copy(skip, z);

  {
    // (0)
    tensor_leaky_relu(z, 0.2f);

    // (1)
    tensor_t* decoder_net_17_aligned_branches_0_net_1_weight = tensor_load_from_file(BIN_PATH"decoder.net.17.aligned.branches.0.net.1.weight.bin", 0);
    tensor_pad(z, 2);
    tensor_conv1d(z, decoder_net_17_aligned_branches_0_net_1_weight, 1, 1);

    // (2)
    tensor_leaky_relu(z, 0.2f);

    // (3)
    tensor_t* decoder_net_17_aligned_branches_0_net_3_weight = tensor_load_from_file(BIN_PATH"decoder.net.17.aligned.branches.0.net.3.weight.bin", 0);
    tensor_conv1d(z, decoder_net_17_aligned_branches_0_net_3_weight, 1, 1);
  }
   
  tensor_tadd(z, skip);

  // (18)
  tensor_copy(skip, z);

  {
    // (0)
    tensor_leaky_relu(z, 0.2f);

    // (1)
    tensor_t* decoder_net_18_aligned_branches_0_net_1_weight = tensor_load_from_file(BIN_PATH"decoder.net.18.aligned.branches.0.net.1.weight.bin", 0);
    tensor_pad(z, 6);
    tensor_conv1d(z, decoder_net_18_aligned_branches_0_net_1_weight, 1, 3);

    // (2)
    tensor_leaky_relu(z, 0.2f);

    // (3)
    tensor_t* decoder_net_18_aligned_branches_0_net_3_weight = tensor_load_from_file(BIN_PATH"decoder.net.18.aligned.branches.0.net.3.weight.bin", 0);
    tensor_conv1d(z, decoder_net_18_aligned_branches_0_net_3_weight, 1, 1);
  }
   
  tensor_tadd(z, skip);

  // (19)
  tensor_copy(skip, z);

  {
    // (0)
    tensor_leaky_relu(z, 0.2f);

    // (1)
    tensor_t* decoder_net_19_aligned_branches_0_net_1_weight = tensor_load_from_file(BIN_PATH"decoder.net.19.aligned.branches.0.net.1.weight.bin", 0);
    tensor_pad(z, 18);
    tensor_conv1d(z, decoder_net_19_aligned_branches_0_net_1_weight, 1, 9);

    // (2)
    tensor_leaky_relu(z, 0.2f);

    // (3)
    tensor_t* decoder_net_19_aligned_branches_0_net_3_weight = tensor_load_from_file(BIN_PATH"decoder.net.19.aligned.branches.0.net.3.weight.bin", 0);
    tensor_conv1d(z, decoder_net_19_aligned_branches_0_net_3_weight, 1, 1);
  }
   
  tensor_tadd(z, skip);

  // (20)
  tensor_t* net_20_alpha = tensor_load_from_file(BIN_PATH"decoder.net.20.alpha.bin", 0);
  tensor_snake(z, net_20_alpha);

  // (21)
  tensor_t* net_21_weight = tensor_load_from_file(BIN_PATH"decoder.net.21.weight.bin", 0);
  tensor_pad(z, 6);
  tensor_conv1d(z, net_21_weight, 1, 1);

  tensor_t* amplitude = skip;

  tensor_split(amplitude, z);
  tensor_sigmoid(amplitude);
  tensor_tmul(z, amplitude);
  tensor_tanh(z);

  post_process_latent(z);
}

#endif // V1_H
