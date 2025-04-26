#ifndef V1_H
#define V1_H

#define V1_BIN_PATH "/Users/lucaayscough/dev/crave/out/bin/"

typedef struct {
  tensor_t* noise;
  tensor_t* latent_pca;
  tensor_t* latent_mean;
  tensor_t* pqmf_inverse_conv_weight;
  tensor_t* net_0_weight;
  tensor_t* net_1_alpha;
  tensor_t* net_2_weight;
  tensor_t* decoder_net_3_aligned_branches_0_net_1_weight;
  tensor_t* decoder_net_3_aligned_branches_0_net_3_weight;
  tensor_t* decoder_net_4_aligned_branches_0_net_1_weight;
  tensor_t* decoder_net_4_aligned_branches_0_net_3_weight;
  tensor_t* net_5_alpha;
  tensor_t* net_6_weight;
  tensor_t* decoder_net_7_aligned_branches_0_net_1_weight;
  tensor_t* decoder_net_7_aligned_branches_0_net_3_weight;
  tensor_t* decoder_net_8_aligned_branches_0_net_1_weight;
  tensor_t* decoder_net_8_aligned_branches_0_net_3_weight;
  tensor_t* decoder_net_9_aligned_branches_0_net_1_weight;
  tensor_t* decoder_net_9_aligned_branches_0_net_3_weight;
  tensor_t* net_10_alpha;
  tensor_t* net_11_weight;
  tensor_t* decoder_net_12_aligned_branches_0_net_1_weight;
  tensor_t* decoder_net_12_aligned_branches_0_net_3_weight;
  tensor_t* decoder_net_13_aligned_branches_0_net_1_weight;
  tensor_t* decoder_net_13_aligned_branches_0_net_3_weight;
  tensor_t* decoder_net_14_aligned_branches_0_net_1_weight;
  tensor_t* decoder_net_14_aligned_branches_0_net_3_weight;
  tensor_t* net_15_alpha;
  tensor_t* net_16_weight;
  tensor_t* decoder_net_17_aligned_branches_0_net_1_weight;
  tensor_t* decoder_net_17_aligned_branches_0_net_3_weight;
  tensor_t* decoder_net_18_aligned_branches_0_net_1_weight;
  tensor_t* decoder_net_18_aligned_branches_0_net_3_weight;
  tensor_t* decoder_net_19_aligned_branches_0_net_1_weight;
  tensor_t* decoder_net_19_aligned_branches_0_net_3_weight;
  tensor_t* net_20_alpha;
  tensor_t* net_21_weight;
  tensor_t* mask;
  tensor_t* skip;
} v1_model_weights_t;

void v1_load_weights(arena_t* arena, v1_model_weights_t* weights) {
  //tensor_list_t* list = tensor_load_from_blob(arena, "weights.bin");
  //weights->noise = tensor_find_in_list(list, "pre_process_latent_noise");
  //weights->latent_pca = tensor_find_in_list(list, "pre_process_latent_latent_pca");
  //weights->latent_mean = tensor_find_in_list(list, "pre_process_latent_latent_mean");
  //weights->pqmf_inverse_conv_weight = tensor_find_in_list(list, "pqmf.inverse_conv.weight");
  //weights->net_0_weight = tensor_find_in_list(list, "decoder.net.0.weight");
  //weights->net_1_alpha = tensor_find_in_list(list, "decoder.net.1.alpha");
  //weights->net_2_weight = tensor_find_in_list(list, "decoder.net.2.weight");
  //weights->decoder_net_3_aligned_branches_0_net_1_weight = tensor_find_in_list(list, "decoder.net.3.aligned.branches.0.net.1.weight");
  //weights->decoder_net_3_aligned_branches_0_net_3_weight = tensor_find_in_list(list, "decoder.net.3.aligned.branches.0.net.3.weight");
  //weights->decoder_net_4_aligned_branches_0_net_1_weight = tensor_find_in_list(list, "decoder.net.4.aligned.branches.0.net.1.weight");
  //weights->decoder_net_4_aligned_branches_0_net_3_weight = tensor_find_in_list(list, "decoder.net.4.aligned.branches.0.net.3.weight");
  //weights->net_5_alpha = tensor_find_in_list(list, "decoder.net.5.alpha");
  //weights->net_6_weight = tensor_find_in_list(list, "decoder.net.6.weight");
  //weights->decoder_net_7_aligned_branches_0_net_1_weight = tensor_find_in_list(list, "decoder.net.7.aligned.branches.0.net.1.weight");
  //weights->decoder_net_7_aligned_branches_0_net_3_weight = tensor_find_in_list(list, "decoder.net.7.aligned.branches.0.net.3.weight");
  //weights->decoder_net_8_aligned_branches_0_net_1_weight = tensor_find_in_list(list, "decoder.net.8.aligned.branches.0.net.1.weight");
  //weights->decoder_net_8_aligned_branches_0_net_3_weight = tensor_find_in_list(list, "decoder.net.8.aligned.branches.0.net.3.weight");
  //weights->decoder_net_9_aligned_branches_0_net_1_weight = tensor_find_in_list(list, "decoder.net.9.aligned.branches.0.net.1.weight");
  //weights->decoder_net_9_aligned_branches_0_net_3_weight = tensor_find_in_list(list, "decoder.net.9.aligned.branches.0.net.3.weight");
  //weights->net_10_alpha = tensor_find_in_list(list, "decoder.net.10.alpha");
  //weights->net_11_weight = tensor_find_in_list(list, "decoder.net.11.weight");
  //weights->decoder_net_12_aligned_branches_0_net_1_weight = tensor_find_in_list(list, "decoder.net.12.aligned.branches.0.net.1.weight");
  //weights->decoder_net_12_aligned_branches_0_net_3_weight = tensor_find_in_list(list, "decoder.net.12.aligned.branches.0.net.3.weight");
  //weights->decoder_net_13_aligned_branches_0_net_1_weight = tensor_find_in_list(list, "decoder.net.13.aligned.branches.0.net.1.weight");
  //weights->decoder_net_13_aligned_branches_0_net_3_weight = tensor_find_in_list(list, "decoder.net.13.aligned.branches.0.net.3.weight");
  //weights->decoder_net_14_aligned_branches_0_net_1_weight = tensor_find_in_list(list, "decoder.net.14.aligned.branches.0.net.1.weight");
  //weights->decoder_net_14_aligned_branches_0_net_3_weight = tensor_find_in_list(list, "decoder.net.14.aligned.branches.0.net.3.weight");
  //weights->net_15_alpha = tensor_find_in_list(list, "decoder.net.15.alpha");
  //weights->net_16_weight = tensor_find_in_list(list, "decoder.net.16.weight");
  //weights->decoder_net_17_aligned_branches_0_net_1_weight = tensor_find_in_list(list, "decoder.net.17.aligned.branches.0.net.1.weight");
  //weights->decoder_net_17_aligned_branches_0_net_3_weight = tensor_find_in_list(list, "decoder.net.17.aligned.branches.0.net.3.weight");
  //weights->decoder_net_18_aligned_branches_0_net_1_weight = tensor_find_in_list(list, "decoder.net.18.aligned.branches.0.net.1.weight");
  //weights->decoder_net_18_aligned_branches_0_net_3_weight = tensor_find_in_list(list, "decoder.net.18.aligned.branches.0.net.3.weight");
  //weights->decoder_net_19_aligned_branches_0_net_1_weight = tensor_find_in_list(list, "decoder.net.19.aligned.branches.0.net.1.weight");
  //weights->decoder_net_19_aligned_branches_0_net_3_weight = tensor_find_in_list(list, "decoder.net.19.aligned.branches.0.net.3.weight");
  //weights->net_20_alpha = tensor_find_in_list(list, "decoder.net.20.alpha");
  //weights->net_21_weight = tensor_find_in_list(list, "decoder.net.21.weight");
 
  weights->noise = tensor_load_from_file(arena, V1_BIN_PATH"pre_process_latent_noise.bin", TENSOR_AUTO_CAP);
  weights->latent_pca = tensor_load_from_file(arena, V1_BIN_PATH"pre_process_latent_latent_pca.bin", TENSOR_AUTO_CAP);
  weights->latent_mean = tensor_load_from_file(arena, V1_BIN_PATH"pre_process_latent_latent_mean.bin", TENSOR_AUTO_CAP);
  weights->pqmf_inverse_conv_weight = tensor_load_from_file(arena, V1_BIN_PATH"pqmf.inverse_conv.weight.bin", TENSOR_AUTO_CAP);
  weights->net_0_weight = tensor_load_from_file(arena, V1_BIN_PATH"decoder.net.0.weight.bin", TENSOR_AUTO_CAP);
  weights->net_1_alpha = tensor_load_from_file(arena, V1_BIN_PATH"decoder.net.1.alpha.bin", TENSOR_AUTO_CAP);
  weights->net_2_weight = tensor_load_from_file(arena, V1_BIN_PATH"decoder.net.2.weight.bin", TENSOR_AUTO_CAP);
  weights->decoder_net_3_aligned_branches_0_net_1_weight = tensor_load_from_file(arena, V1_BIN_PATH"decoder.net.3.aligned.branches.0.net.1.weight.bin", TENSOR_AUTO_CAP);
  weights->decoder_net_3_aligned_branches_0_net_3_weight = tensor_load_from_file(arena, V1_BIN_PATH"decoder.net.3.aligned.branches.0.net.3.weight.bin", TENSOR_AUTO_CAP);
  weights->decoder_net_4_aligned_branches_0_net_1_weight = tensor_load_from_file(arena, V1_BIN_PATH"decoder.net.4.aligned.branches.0.net.1.weight.bin", TENSOR_AUTO_CAP);
  weights->decoder_net_4_aligned_branches_0_net_3_weight = tensor_load_from_file(arena, V1_BIN_PATH"decoder.net.4.aligned.branches.0.net.3.weight.bin", TENSOR_AUTO_CAP);
  weights->net_5_alpha = tensor_load_from_file(arena, V1_BIN_PATH"decoder.net.5.alpha.bin", TENSOR_AUTO_CAP);
  weights->net_6_weight = tensor_load_from_file(arena, V1_BIN_PATH"decoder.net.6.weight.bin", TENSOR_AUTO_CAP);
  weights->decoder_net_7_aligned_branches_0_net_1_weight = tensor_load_from_file(arena, V1_BIN_PATH"decoder.net.7.aligned.branches.0.net.1.weight.bin", TENSOR_AUTO_CAP);
  weights->decoder_net_7_aligned_branches_0_net_3_weight = tensor_load_from_file(arena, V1_BIN_PATH"decoder.net.7.aligned.branches.0.net.3.weight.bin", TENSOR_AUTO_CAP);
  weights->decoder_net_8_aligned_branches_0_net_1_weight = tensor_load_from_file(arena, V1_BIN_PATH"decoder.net.8.aligned.branches.0.net.1.weight.bin", TENSOR_AUTO_CAP);
  weights->decoder_net_8_aligned_branches_0_net_3_weight = tensor_load_from_file(arena, V1_BIN_PATH"decoder.net.8.aligned.branches.0.net.3.weight.bin", TENSOR_AUTO_CAP);
  weights->decoder_net_9_aligned_branches_0_net_1_weight = tensor_load_from_file(arena, V1_BIN_PATH"decoder.net.9.aligned.branches.0.net.1.weight.bin", TENSOR_AUTO_CAP);
  weights->decoder_net_9_aligned_branches_0_net_3_weight = tensor_load_from_file(arena, V1_BIN_PATH"decoder.net.9.aligned.branches.0.net.3.weight.bin", TENSOR_AUTO_CAP);
  weights->net_10_alpha = tensor_load_from_file(arena, V1_BIN_PATH"decoder.net.10.alpha.bin", TENSOR_AUTO_CAP);
  weights->net_11_weight = tensor_load_from_file(arena, V1_BIN_PATH"decoder.net.11.weight.bin", TENSOR_AUTO_CAP);
  weights->decoder_net_12_aligned_branches_0_net_1_weight = tensor_load_from_file(arena, V1_BIN_PATH"decoder.net.12.aligned.branches.0.net.1.weight.bin", TENSOR_AUTO_CAP);
  weights->decoder_net_12_aligned_branches_0_net_3_weight = tensor_load_from_file(arena, V1_BIN_PATH"decoder.net.12.aligned.branches.0.net.3.weight.bin", TENSOR_AUTO_CAP);
  weights->decoder_net_13_aligned_branches_0_net_1_weight = tensor_load_from_file(arena, V1_BIN_PATH"decoder.net.13.aligned.branches.0.net.1.weight.bin", TENSOR_AUTO_CAP);
  weights->decoder_net_13_aligned_branches_0_net_3_weight = tensor_load_from_file(arena, V1_BIN_PATH"decoder.net.13.aligned.branches.0.net.3.weight.bin", TENSOR_AUTO_CAP);
  weights->decoder_net_14_aligned_branches_0_net_1_weight = tensor_load_from_file(arena, V1_BIN_PATH"decoder.net.14.aligned.branches.0.net.1.weight.bin", TENSOR_AUTO_CAP);
  weights->decoder_net_14_aligned_branches_0_net_3_weight = tensor_load_from_file(arena, V1_BIN_PATH"decoder.net.14.aligned.branches.0.net.3.weight.bin", TENSOR_AUTO_CAP);
  weights->net_15_alpha = tensor_load_from_file(arena, V1_BIN_PATH"decoder.net.15.alpha.bin", TENSOR_AUTO_CAP);
  weights->net_16_weight = tensor_load_from_file(arena, V1_BIN_PATH"decoder.net.16.weight.bin", TENSOR_AUTO_CAP);
  weights->decoder_net_17_aligned_branches_0_net_1_weight = tensor_load_from_file(arena, V1_BIN_PATH"decoder.net.17.aligned.branches.0.net.1.weight.bin", TENSOR_AUTO_CAP);
  weights->decoder_net_17_aligned_branches_0_net_3_weight = tensor_load_from_file(arena, V1_BIN_PATH"decoder.net.17.aligned.branches.0.net.3.weight.bin", TENSOR_AUTO_CAP);
  weights->decoder_net_18_aligned_branches_0_net_1_weight = tensor_load_from_file(arena, V1_BIN_PATH"decoder.net.18.aligned.branches.0.net.1.weight.bin", TENSOR_AUTO_CAP);
  weights->decoder_net_18_aligned_branches_0_net_3_weight = tensor_load_from_file(arena, V1_BIN_PATH"decoder.net.18.aligned.branches.0.net.3.weight.bin", TENSOR_AUTO_CAP);
  weights->decoder_net_19_aligned_branches_0_net_1_weight = tensor_load_from_file(arena, V1_BIN_PATH"decoder.net.19.aligned.branches.0.net.1.weight.bin", TENSOR_AUTO_CAP);
  weights->decoder_net_19_aligned_branches_0_net_3_weight = tensor_load_from_file(arena, V1_BIN_PATH"decoder.net.19.aligned.branches.0.net.3.weight.bin", TENSOR_AUTO_CAP);
  weights->net_20_alpha = tensor_load_from_file(arena, V1_BIN_PATH"decoder.net.20.alpha.bin", TENSOR_AUTO_CAP);
  weights->net_21_weight = tensor_load_from_file(arena, V1_BIN_PATH"decoder.net.21.weight.bin", TENSOR_AUTO_CAP);

  weights->skip = tensor_create(arena, U32_TPL(1), 2 * 1024);
  weights->mask = tensor_create(arena, U32_TPL(1, 16, 128), TENSOR_AUTO_CAP);
  tensor_fill(weights->mask, 1.f);

  size_t channels = weights->mask->dims[1];
  size_t len = weights->mask->dims[2];
  
  for (size_t i = 1; i < channels; i += 2) {
    for (size_t j = 0; j < len; j += 2) {
      size_t idx = i * len + j; 
      weights->mask->data[idx] = -1; 
    }
  }

  tensor_unsqueeze(weights->latent_pca, weights->latent_pca->rank);
  tensor_unsqueeze(weights->latent_mean, 0);
  tensor_unsqueeze(weights->latent_mean, weights->latent_mean->rank);
}

void v1_decode(tensor_t* z, v1_model_weights_t* w) {
  tensor_transpose(w->latent_pca, 0, 1);
  tensor_cat(z, w->noise, 1);
  tensor_conv1d(z, w->latent_pca, 1, 1);
  tensor_tadd(z, w->latent_mean);

  // (0)
  tensor_pad(z, 2);
  tensor_conv1d(z, w->net_0_weight, 1, 1);

  // (1)
  tensor_snake(z, w->net_1_alpha);

  // (2)
  tensor_pad(z, 1);
  tensor_conv_transpose1d(z, w->net_2_weight, 2, 1);
  tensor_trunc(z, 2, 2);

  // (3)
  tensor_copy(w->skip, z);

  {
    // (0)
    tensor_leaky_relu(z, 0.2f);

    // (1)
    tensor_pad(z, 2);
    tensor_conv1d(z, w->decoder_net_3_aligned_branches_0_net_1_weight, 1, 1);

    // (2)
    tensor_leaky_relu(z, 0.2f);

    // (3)
    tensor_conv1d(z, w->decoder_net_3_aligned_branches_0_net_3_weight, 1, 1);
  }

  tensor_tadd(z, w->skip);

  // (4)
  tensor_copy(w->skip, z);

  {
    // (0)
    tensor_leaky_relu(z, 0.2f);

    // (1)
    tensor_pad(z, 6);
    tensor_conv1d(z, w->decoder_net_4_aligned_branches_0_net_1_weight, 1, 3);

    // (2)
    tensor_leaky_relu(z, 0.2f);

    // (3)
    tensor_conv1d(z, w->decoder_net_4_aligned_branches_0_net_3_weight, 1, 1);
  }

  tensor_tadd(z, w->skip);

  // (5)
  tensor_snake(z, w->net_5_alpha);

  // (6)
  tensor_pad(z, 1);
  tensor_conv_transpose1d(z, w->net_6_weight, 4, 1);
  tensor_trunc(z, 4, 4);

  // (7)
  tensor_copy(w->skip, z);

  {
    // (0)
    tensor_leaky_relu(z, 0.2f);

    // (1)
    tensor_pad(z, 2);
    tensor_conv1d(z, w->decoder_net_7_aligned_branches_0_net_1_weight, 1, 1);

    // (2)
    tensor_leaky_relu(z, 0.2f);

    // (3)
    tensor_conv1d(z, w->decoder_net_7_aligned_branches_0_net_3_weight, 1, 1);
  }

  tensor_tadd(z, w->skip);
  // (8)
  tensor_copy(w->skip, z);

  {
    // (0)
    tensor_leaky_relu(z, 0.2f);

    // (1)
    tensor_pad(z, 6);
    tensor_conv1d(z, w->decoder_net_8_aligned_branches_0_net_1_weight, 1, 3);

    // (2)
    tensor_leaky_relu(z, 0.2f);

    // (3)
    tensor_conv1d(z, w->decoder_net_8_aligned_branches_0_net_3_weight, 1, 1);
  }

  tensor_tadd(z, w->skip);

  // (9)
  tensor_copy(w->skip, z);

  {
    // (0)
    tensor_leaky_relu(z, 0.2f);

    // (1)
    tensor_pad(z, 18);
    tensor_conv1d(z, w->decoder_net_9_aligned_branches_0_net_1_weight, 1, 9);

    // (2)
    tensor_leaky_relu(z, 0.2f);

    // (3)
    tensor_conv1d(z, w->decoder_net_9_aligned_branches_0_net_3_weight, 1, 1);
  }

  tensor_tadd(z, w->skip);

  // (10)
  tensor_snake(z, w->net_10_alpha);

  // (11)
  tensor_pad(z, 1);
  tensor_conv_transpose1d(z, w->net_11_weight, 4, 1);
  tensor_trunc(z, 4, 4);

  // (12)
  tensor_copy(w->skip, z);

  {
    // (0)
    tensor_leaky_relu(z, 0.2f);

    // (1)
    tensor_pad(z, 2);
    tensor_conv1d(z, w->decoder_net_12_aligned_branches_0_net_1_weight, 1, 1);

    // (2)
    tensor_leaky_relu(z, 0.2f);

    // (3)
    tensor_conv1d(z, w->decoder_net_12_aligned_branches_0_net_3_weight, 1, 1);
  }

  tensor_tadd(z, w->skip);

  // (13)
  tensor_copy(w->skip, z);

  {
    // (0)
    tensor_leaky_relu(z, 0.2f);

    // (1)
    tensor_pad(z, 6);
    tensor_conv1d(z, w->decoder_net_13_aligned_branches_0_net_1_weight, 1, 3);

    // (2)
    tensor_leaky_relu(z, 0.2f);

    // (3)
    tensor_conv1d(z, w->decoder_net_13_aligned_branches_0_net_3_weight, 1, 1);
  }
   
  tensor_tadd(z, w->skip);

  // (14)
  tensor_copy(w->skip, z);

  {
    // (0)
    tensor_leaky_relu(z, 0.2f);

    // (1)
    tensor_pad(z, 18);
    tensor_conv1d(z, w->decoder_net_14_aligned_branches_0_net_1_weight, 1, 9);

    // (2)
    tensor_leaky_relu(z, 0.2f);

    // (3)
    tensor_conv1d(z, w->decoder_net_14_aligned_branches_0_net_3_weight, 1, 1);
  }
   
  tensor_tadd(z, w->skip);

  // (15)
  tensor_snake(z, w->net_15_alpha);

  // (16)
  tensor_pad(z, 1);
  tensor_conv_transpose1d(z, w->net_16_weight, 4, 1);
  tensor_trunc(z, 4, 4);

  // (17)
  tensor_copy(w->skip, z);

  {
    // (0)
    tensor_leaky_relu(z, 0.2f);

    // (1)
    tensor_pad(z, 2);
    tensor_conv1d(z, w->decoder_net_17_aligned_branches_0_net_1_weight, 1, 1);

    // (2)
    tensor_leaky_relu(z, 0.2f);

    // (3)
    tensor_conv1d(z, w->decoder_net_17_aligned_branches_0_net_3_weight, 1, 1);
  }
   
  tensor_tadd(z, w->skip);

  // (18)
  tensor_copy(w->skip, z);

  {
    // (0)
    tensor_leaky_relu(z, 0.2f);

    // (1)
    tensor_pad(z, 6);
    tensor_conv1d(z, w->decoder_net_18_aligned_branches_0_net_1_weight, 1, 3);

    // (2)
    tensor_leaky_relu(z, 0.2f);

    // (3)
    tensor_conv1d(z, w->decoder_net_18_aligned_branches_0_net_3_weight, 1, 1);
  }
   
  tensor_tadd(z, w->skip);

  // (19)
  tensor_copy(w->skip, z);

  {
    // (0)
    tensor_leaky_relu(z, 0.2f);

    // (1)
    tensor_pad(z, 18);
    tensor_conv1d(z, w->decoder_net_19_aligned_branches_0_net_1_weight, 1, 9);

    // (2)
    tensor_leaky_relu(z, 0.2f);

    // (3)
    tensor_conv1d(z, w->decoder_net_19_aligned_branches_0_net_3_weight, 1, 1);
  }
   
  tensor_tadd(z, w->skip);

  // (20)
  tensor_snake(z, w->net_20_alpha);

  // (21)
  tensor_pad(z, 6);
  tensor_conv1d(z, w->net_21_weight, 1, 1);

  tensor_t* amplitude = w->skip;

  tensor_split(amplitude, z);
  tensor_sigmoid(amplitude);
  tensor_tmul(z, amplitude);
  tensor_tanh(z);

  tensor_reshape(z, U32_TPL(1, 16, 128)); 
  tensor_tmul(z, w->mask);
  tensor_pad(z, 32);
  tensor_conv1d(z, w->pqmf_inverse_conv_weight, 1, 1);
  tensor_mul(z, 16);
  tensor_flip(z, 1);
  tensor_permute(z, U32_TPL(0, 2, 1));
  tensor_reshape(z, U32_TPL(1, 128, 1, 16));
  tensor_permute(z, U32_TPL(0, 2, 1, 3));
  tensor_reshape(z, U32_TPL(1, 1, 2048));
}

#endif // V1_H
