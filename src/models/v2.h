#ifndef V2_H
#define V2_H

#define V2_BIN_PATH "/Users/lucaayscough/dev/crave/out/bin2/"

void v2_cache_slice(tensor_t* cache, tensor_t* input) {
  size_t input_size = crv_get_tensor_last_dim_size(input);
  size_t cache_size = crv_get_tensor_last_dim_size(cache);

  size_t start = input_size - cache_size;
  size_t cpy_count = input_size - start;

  DO_INTERNAL(
    crv_validate_tensor(input);
    crv_validate_tensor(cache);

    assert(input->rank == cache->rank);
    assert(input->dims[0] == cache->dims[0]);
    assert(input->dims[1] == cache->dims[1]);
    assert(input_size >= cache_size);
  );

  size_t strides[CRV_MAX_RANK];
  crv_get_tensor_strides(input, &strides[0]);

  size_t stride = strides[1];
  size_t dims = input->dims[1];

  float* x = input->data + start;
  float* y = cache->data;

  for (size_t i = 0; i < dims; ++i) {
    memcpy(y, x, cpy_count * sizeof(float));
    x += stride;
    y += cpy_count; 
  }
}

void v2_cached_pad(tensor_t* input, tensor_t* cache) {
  size_t cache_size = crv_get_tensor_last_dim_size(cache);
  tensor_cat(input, cache, crv_get_tensor_last_dim_index(input), CRV_FRONT);
  v2_cache_slice(cache, input);
  tensor_trunc(input, 0, cache_size);
}

void v2_cached_conv1d(tensor_t* input, tensor_t* weights, tensor_t* cache, size_t stride, size_t dilation) {
  tensor_cat(input, cache, crv_get_tensor_last_dim_index(input), CRV_FRONT);
  v2_cache_slice(cache, input);
  tensor_conv1d(input, weights, stride, dilation);
}

void v2_cached_conv_transpose1d(tensor_t* input, tensor_t* weights, tensor_t* cache, size_t stride, size_t dilation) {
  tensor_conv_transpose1d(input, weights, stride, dilation);

  size_t cache_size = crv_get_tensor_last_dim_size(cache);

  DO_INTERNAL(
    crv_validate_tensor(input);
    crv_validate_tensor(cache);

    assert(input->rank == cache->rank);
    assert(input->dims[0] == cache->dims[0]);
    assert(input->dims[1] == cache->dims[1]);

    size_t input_size = crv_get_tensor_last_dim_size(input);
    assert(input_size >= cache_size);
  );

  size_t strides[CRV_MAX_RANK];
  crv_get_tensor_strides(input, &strides[0]);

  size_t input_stride = strides[1];
  size_t dims = input->dims[1];

  float* x = input->data;
  float* y = cache->data;

  for (size_t i = 0; i < dims; ++i) {
    for (size_t j = 0; j < cache_size; ++j) {
      x[j] += y[j];
    }

    x += input_stride;
    y += cache_size; 
  }

  v2_cache_slice(cache, input);
  tensor_trunc(input, 0, cache_size);
}

typedef struct {
  tensor_t* noise;
  tensor_t* latent_pca;
  tensor_t* latent_mean;
  tensor_t* decoder_net_0_cache_pad;
  tensor_t* decoder_net_0_weight;
  tensor_t* decoder_net_1_alpha;
  tensor_t* decoder_net_2_cache;
  tensor_t* decoder_net_2_weight;
  tensor_t* decoder_net_3_aligned_paddings_1_pad;
  tensor_t* decoder_net_3_aligned_branches_0_net_1_cache_pad;
  tensor_t* decoder_net_3_aligned_branches_0_net_1_weight;
  tensor_t* decoder_net_3_aligned_branches_0_net_3_weight;
  tensor_t* decoder_net_4_aligned_paddings_1_pad;
  tensor_t* decoder_net_4_aligned_branches_0_net_1_cache_pad;
  tensor_t* decoder_net_4_aligned_branches_0_net_1_weight;
  tensor_t* decoder_net_4_aligned_branches_0_net_3_weight;
  tensor_t* decoder_net_5_alpha;
  tensor_t* decoder_net_6_cache;
  tensor_t* decoder_net_6_weight;
  tensor_t* decoder_net_7_aligned_paddings_1_pad;
  tensor_t* decoder_net_7_aligned_branches_0_net_1_cache_pad;
  tensor_t* decoder_net_7_aligned_branches_0_net_1_weight;
  tensor_t* decoder_net_7_aligned_branches_0_net_3_weight;
  tensor_t* decoder_net_8_aligned_paddings_1_pad;
  tensor_t* decoder_net_8_aligned_branches_0_net_1_cache_pad;
  tensor_t* decoder_net_8_aligned_branches_0_net_1_weight;
  tensor_t* decoder_net_8_aligned_branches_0_net_3_weight;
  tensor_t* decoder_net_9_aligned_paddings_1_pad;
  tensor_t* decoder_net_9_aligned_branches_0_net_1_cache_pad;
  tensor_t* decoder_net_9_aligned_branches_0_net_1_weight;
  tensor_t* decoder_net_9_aligned_branches_0_net_3_weight;
  tensor_t* decoder_net_10_alpha;
  tensor_t* decoder_net_11_cache;
  tensor_t* decoder_net_11_weight;
  tensor_t* decoder_net_12_aligned_paddings_1_pad;
  tensor_t* decoder_net_12_aligned_branches_0_net_1_cache_pad;
  tensor_t* decoder_net_12_aligned_branches_0_net_1_weight;
  tensor_t* decoder_net_12_aligned_branches_0_net_3_weight;
  tensor_t* decoder_net_13_aligned_paddings_1_pad;
  tensor_t* decoder_net_13_aligned_branches_0_net_1_cache_pad;
  tensor_t* decoder_net_13_aligned_branches_0_net_1_weight;
  tensor_t* decoder_net_13_aligned_branches_0_net_3_weight;
  tensor_t* decoder_net_14_aligned_paddings_1_pad;
  tensor_t* decoder_net_14_aligned_branches_0_net_1_cache_pad;
  tensor_t* decoder_net_14_aligned_branches_0_net_1_weight;
  tensor_t* decoder_net_14_aligned_branches_0_net_3_weight;
  tensor_t* decoder_net_15_alpha;
  tensor_t* decoder_net_16_cache;
  tensor_t* decoder_net_16_weight;
  tensor_t* decoder_net_17_aligned_paddings_1_pad;
  tensor_t* decoder_net_17_aligned_branches_0_net_1_cache_pad;
  tensor_t* decoder_net_17_aligned_branches_0_net_1_weight;
  tensor_t* decoder_net_17_aligned_branches_0_net_3_weight;
  tensor_t* decoder_net_18_aligned_paddings_1_pad;
  tensor_t* decoder_net_18_aligned_branches_0_net_1_cache_pad;
  tensor_t* decoder_net_18_aligned_branches_0_net_1_weight;
  tensor_t* decoder_net_18_aligned_branches_0_net_3_weight;
  tensor_t* decoder_net_19_aligned_paddings_1_pad;
  tensor_t* decoder_net_19_aligned_branches_0_net_1_cache_pad;
  tensor_t* decoder_net_19_aligned_branches_0_net_1_weight;
  tensor_t* decoder_net_19_aligned_branches_0_net_3_weight;
  tensor_t* decoder_net_20_alpha;
  tensor_t* decoder_noise_module_net_0_cache_pad;
  tensor_t* decoder_noise_module_net_0_weight;
  tensor_t* decoder_noise_module_net_1_alpha;
  tensor_t* decoder_noise_module_net_2_cache_pad;
  tensor_t* decoder_noise_module_net_2_weight;
  tensor_t* decoder_noise_module_net_3_alpha;
  tensor_t* decoder_noise_module_net_4_cache_pad;
  tensor_t* decoder_noise_module_net_4_weight;
  tensor_t* decoder_waveform_module_cache_pad;
  tensor_t* decoder_waveform_module_weight;
  tensor_t* pqmf_inverse_conv_cache_pad;
  tensor_t* pqmf_inverse_conv_weight;
  tensor_t* mask;
  tensor_t* skip;
  tensor_t* zeros;
  tensor_t* hann;
  tensor_t* ir_noise;
  tensor_t* scratch_1;
  tensor_t* scratch_2;
} v2_model_t;

void v2_load_weights(arena_t* arena, v2_model_t* w, tensor_list_t* list) {
  // TODO(luca): Add error handling.
  w->noise                                              = tensor_find_in_list(list, "pre_process_latent_noise");
  w->latent_pca                                         = tensor_find_in_list(list, "latent_pca");
  w->latent_mean                                        = tensor_find_in_list(list, "latent_mean");
  w->decoder_net_0_weight                               = tensor_find_in_list(list, "decoder.net.0.weight");
  w->decoder_net_0_cache_pad                            = tensor_find_in_list(list, "decoder.net.0.cache.pad");
  w->decoder_net_1_alpha                                = tensor_find_in_list(list, "decoder.net.1.alpha");
  w->decoder_net_2_cache                                = tensor_find_in_list(list, "decoder.net.2.cache");
  w->decoder_net_2_weight                               = tensor_find_in_list(list, "decoder.net.2.weight");
  w->decoder_net_3_aligned_paddings_1_pad               = tensor_find_in_list(list, "decoder.net.3.aligned.paddings.1.pad");
  w->decoder_net_3_aligned_branches_0_net_1_cache_pad   = tensor_find_in_list(list, "decoder.net.3.aligned.branches.0.net.1.cache.pad");
  w->decoder_net_3_aligned_branches_0_net_1_weight      = tensor_find_in_list(list, "decoder.net.3.aligned.branches.0.net.1.weight");
  w->decoder_net_3_aligned_branches_0_net_3_weight      = tensor_find_in_list(list, "decoder.net.3.aligned.branches.0.net.3.weight");
  w->decoder_net_4_aligned_paddings_1_pad               = tensor_find_in_list(list, "decoder.net.4.aligned.paddings.1.pad");
  w->decoder_net_4_aligned_branches_0_net_1_cache_pad   = tensor_find_in_list(list, "decoder.net.4.aligned.branches.0.net.1.cache.pad");
  w->decoder_net_4_aligned_branches_0_net_1_weight      = tensor_find_in_list(list, "decoder.net.4.aligned.branches.0.net.1.weight");
  w->decoder_net_4_aligned_branches_0_net_3_weight      = tensor_find_in_list(list, "decoder.net.4.aligned.branches.0.net.3.weight");
  w->decoder_net_5_alpha                                = tensor_find_in_list(list, "decoder.net.5.alpha");
  w->decoder_net_6_cache                                = tensor_find_in_list(list, "decoder.net.6.cache");
  w->decoder_net_6_weight                               = tensor_find_in_list(list, "decoder.net.6.weight");
  w->decoder_net_7_aligned_paddings_1_pad               = tensor_find_in_list(list, "decoder.net.7.aligned.paddings.1.pad");
  w->decoder_net_7_aligned_branches_0_net_1_cache_pad   = tensor_find_in_list(list, "decoder.net.7.aligned.branches.0.net.1.cache.pad");
  w->decoder_net_7_aligned_branches_0_net_1_weight      = tensor_find_in_list(list, "decoder.net.7.aligned.branches.0.net.1.weight");
  w->decoder_net_7_aligned_branches_0_net_3_weight      = tensor_find_in_list(list, "decoder.net.7.aligned.branches.0.net.3.weight");
  w->decoder_net_8_aligned_paddings_1_pad               = tensor_find_in_list(list, "decoder.net.8.aligned.paddings.1.pad");
  w->decoder_net_8_aligned_branches_0_net_1_cache_pad   = tensor_find_in_list(list, "decoder.net.8.aligned.branches.0.net.1.cache.pad");
  w->decoder_net_8_aligned_branches_0_net_1_weight      = tensor_find_in_list(list, "decoder.net.8.aligned.branches.0.net.1.weight");
  w->decoder_net_8_aligned_branches_0_net_3_weight      = tensor_find_in_list(list, "decoder.net.8.aligned.branches.0.net.3.weight");
  w->decoder_net_9_aligned_paddings_1_pad               = tensor_find_in_list(list, "decoder.net.9.aligned.paddings.1.pad");
  w->decoder_net_9_aligned_branches_0_net_1_cache_pad   = tensor_find_in_list(list, "decoder.net.9.aligned.branches.0.net.1.cache.pad");
  w->decoder_net_9_aligned_branches_0_net_1_weight      = tensor_find_in_list(list, "decoder.net.9.aligned.branches.0.net.1.weight");
  w->decoder_net_9_aligned_branches_0_net_3_weight      = tensor_find_in_list(list, "decoder.net.9.aligned.branches.0.net.3.weight");
  w->decoder_net_10_alpha                               = tensor_find_in_list(list, "decoder.net.10.alpha");
  w->decoder_net_11_cache                               = tensor_find_in_list(list, "decoder.net.11.cache");
  w->decoder_net_11_weight                              = tensor_find_in_list(list, "decoder.net.11.weight");
  w->decoder_net_12_aligned_paddings_1_pad              = tensor_find_in_list(list, "decoder.net.12.aligned.paddings.1.pad");
  w->decoder_net_12_aligned_branches_0_net_1_cache_pad  = tensor_find_in_list(list, "decoder.net.12.aligned.branches.0.net.1.cache.pad");
  w->decoder_net_12_aligned_branches_0_net_1_weight     = tensor_find_in_list(list, "decoder.net.12.aligned.branches.0.net.1.weight");
  w->decoder_net_12_aligned_branches_0_net_3_weight     = tensor_find_in_list(list, "decoder.net.12.aligned.branches.0.net.3.weight");
  w->decoder_net_13_aligned_paddings_1_pad              = tensor_find_in_list(list, "decoder.net.13.aligned.paddings.1.pad");
  w->decoder_net_13_aligned_branches_0_net_1_cache_pad  = tensor_find_in_list(list, "decoder.net.13.aligned.branches.0.net.1.cache.pad");
  w->decoder_net_13_aligned_branches_0_net_1_weight     = tensor_find_in_list(list, "decoder.net.13.aligned.branches.0.net.1.weight");
  w->decoder_net_13_aligned_branches_0_net_3_weight     = tensor_find_in_list(list, "decoder.net.13.aligned.branches.0.net.3.weight");
  w->decoder_net_14_aligned_paddings_1_pad              = tensor_find_in_list(list, "decoder.net.14.aligned.paddings.1.pad");
  w->decoder_net_14_aligned_branches_0_net_1_cache_pad  = tensor_find_in_list(list, "decoder.net.14.aligned.branches.0.net.1.cache.pad");
  w->decoder_net_14_aligned_branches_0_net_1_weight     = tensor_find_in_list(list, "decoder.net.14.aligned.branches.0.net.1.weight");
  w->decoder_net_14_aligned_branches_0_net_3_weight     = tensor_find_in_list(list, "decoder.net.14.aligned.branches.0.net.3.weight");
  w->decoder_net_15_alpha                               = tensor_find_in_list(list, "decoder.net.15.alpha");
  w->decoder_net_16_cache                               = tensor_find_in_list(list, "decoder.net.16.cache");
  w->decoder_net_16_weight                              = tensor_find_in_list(list, "decoder.net.16.weight");
  w->decoder_net_17_aligned_paddings_1_pad              = tensor_find_in_list(list, "decoder.net.17.aligned.paddings.1.pad");
  w->decoder_net_17_aligned_branches_0_net_1_cache_pad  = tensor_find_in_list(list, "decoder.net.17.aligned.branches.0.net.1.cache.pad");
  w->decoder_net_17_aligned_branches_0_net_1_weight     = tensor_find_in_list(list, "decoder.net.17.aligned.branches.0.net.1.weight");
  w->decoder_net_17_aligned_branches_0_net_3_weight     = tensor_find_in_list(list, "decoder.net.17.aligned.branches.0.net.3.weight");
  w->decoder_net_18_aligned_paddings_1_pad              = tensor_find_in_list(list, "decoder.net.18.aligned.paddings.1.pad");
  w->decoder_net_18_aligned_branches_0_net_1_cache_pad  = tensor_find_in_list(list, "decoder.net.18.aligned.branches.0.net.1.cache.pad");
  w->decoder_net_18_aligned_branches_0_net_1_weight     = tensor_find_in_list(list, "decoder.net.18.aligned.branches.0.net.1.weight");
  w->decoder_net_18_aligned_branches_0_net_3_weight     = tensor_find_in_list(list, "decoder.net.18.aligned.branches.0.net.3.weight");
  w->decoder_net_19_aligned_paddings_1_pad              = tensor_find_in_list(list, "decoder.net.19.aligned.paddings.1.pad");
  w->decoder_net_19_aligned_branches_0_net_1_cache_pad  = tensor_find_in_list(list, "decoder.net.19.aligned.branches.0.net.1.cache.pad");
  w->decoder_net_19_aligned_branches_0_net_1_weight     = tensor_find_in_list(list, "decoder.net.19.aligned.branches.0.net.1.weight");
  w->decoder_net_19_aligned_branches_0_net_3_weight     = tensor_find_in_list(list, "decoder.net.19.aligned.branches.0.net.3.weight");
  w->decoder_net_20_alpha                               = tensor_find_in_list(list, "decoder.net.20.alpha");
  w->decoder_noise_module_net_0_cache_pad               = tensor_find_in_list(list, "decoder.noise_module.net.0.cache.pad");
  w->decoder_noise_module_net_0_weight                  = tensor_find_in_list(list, "decoder.noise_module.net.0.weight");
  w->decoder_noise_module_net_1_alpha                   = tensor_find_in_list(list, "decoder.noise_module.net.1.alpha");
  w->decoder_noise_module_net_2_cache_pad               = tensor_find_in_list(list, "decoder.noise_module.net.2.cache.pad");
  w->decoder_noise_module_net_2_weight                  = tensor_find_in_list(list, "decoder.noise_module.net.2.weight");
  w->decoder_noise_module_net_3_alpha                   = tensor_find_in_list(list, "decoder.noise_module.net.3.alpha");
  w->decoder_noise_module_net_4_cache_pad               = tensor_find_in_list(list, "decoder.noise_module.net.4.cache.pad");
  w->decoder_noise_module_net_4_weight                  = tensor_find_in_list(list, "decoder.noise_module.net.4.weight");
  w->decoder_waveform_module_cache_pad                  = tensor_find_in_list(list, "decoder.waveform_module.cache.pad");
  w->decoder_waveform_module_weight                     = tensor_find_in_list(list, "decoder.waveform_module.weight");
  w->pqmf_inverse_conv_cache_pad                        = tensor_find_in_list(list, "pqmf.inverse_conv.cache.pad");
  w->pqmf_inverse_conv_weight                           = tensor_find_in_list(list, "pqmf.inverse_conv.weight");
  w->ir_noise                                           = tensor_find_in_list(list, "ir_noise");

  assert(w->noise != NULL);
  assert(w->latent_pca != NULL);
  assert(w->latent_mean != NULL);
  assert(w->decoder_net_0_weight != NULL);
  assert(w->decoder_net_0_cache_pad != NULL);
  assert(w->decoder_net_1_alpha != NULL);
  assert(w->decoder_net_2_cache != NULL);
  assert(w->decoder_net_2_weight != NULL);
  assert(w->decoder_net_3_aligned_paddings_1_pad != NULL);
  assert(w->decoder_net_3_aligned_branches_0_net_1_cache_pad != NULL);
  assert(w->decoder_net_3_aligned_branches_0_net_1_weight != NULL);
  assert(w->decoder_net_3_aligned_branches_0_net_3_weight != NULL);
  assert(w->decoder_net_4_aligned_paddings_1_pad != NULL);
  assert(w->decoder_net_4_aligned_branches_0_net_1_cache_pad != NULL);
  assert(w->decoder_net_4_aligned_branches_0_net_1_weight != NULL);
  assert(w->decoder_net_4_aligned_branches_0_net_3_weight != NULL);
  assert(w->decoder_net_5_alpha != NULL);
  assert(w->decoder_net_6_cache != NULL);
  assert(w->decoder_net_6_weight != NULL);
  assert(w->decoder_net_7_aligned_paddings_1_pad != NULL);
  assert(w->decoder_net_7_aligned_branches_0_net_1_cache_pad != NULL);
  assert(w->decoder_net_7_aligned_branches_0_net_1_weight != NULL);
  assert(w->decoder_net_7_aligned_branches_0_net_3_weight != NULL);
  assert(w->decoder_net_8_aligned_paddings_1_pad != NULL);
  assert(w->decoder_net_8_aligned_branches_0_net_1_cache_pad != NULL);
  assert(w->decoder_net_8_aligned_branches_0_net_1_weight != NULL);
  assert(w->decoder_net_8_aligned_branches_0_net_3_weight != NULL);
  assert(w->decoder_net_9_aligned_paddings_1_pad != NULL);
  assert(w->decoder_net_9_aligned_branches_0_net_1_cache_pad != NULL);
  assert(w->decoder_net_9_aligned_branches_0_net_1_weight != NULL);
  assert(w->decoder_net_9_aligned_branches_0_net_3_weight != NULL);
  assert(w->decoder_net_10_alpha != NULL);
  assert(w->decoder_net_11_cache != NULL);
  assert(w->decoder_net_11_weight != NULL);
  assert(w->decoder_net_12_aligned_paddings_1_pad != NULL);
  assert(w->decoder_net_12_aligned_branches_0_net_1_cache_pad != NULL);
  assert(w->decoder_net_12_aligned_branches_0_net_1_weight != NULL);
  assert(w->decoder_net_12_aligned_branches_0_net_3_weight != NULL);
  assert(w->decoder_net_13_aligned_paddings_1_pad != NULL);
  assert(w->decoder_net_13_aligned_branches_0_net_1_cache_pad != NULL);
  assert(w->decoder_net_13_aligned_branches_0_net_1_weight != NULL);
  assert(w->decoder_net_13_aligned_branches_0_net_3_weight != NULL);
  assert(w->decoder_net_14_aligned_paddings_1_pad != NULL);
  assert(w->decoder_net_14_aligned_branches_0_net_1_cache_pad != NULL);
  assert(w->decoder_net_14_aligned_branches_0_net_1_weight != NULL);
  assert(w->decoder_net_14_aligned_branches_0_net_3_weight != NULL);
  assert(w->decoder_net_15_alpha != NULL);
  assert(w->decoder_net_16_cache != NULL);
  assert(w->decoder_net_16_weight != NULL);
  assert(w->decoder_net_17_aligned_paddings_1_pad != NULL);
  assert(w->decoder_net_17_aligned_branches_0_net_1_cache_pad != NULL);
  assert(w->decoder_net_17_aligned_branches_0_net_1_weight != NULL);
  assert(w->decoder_net_17_aligned_branches_0_net_3_weight != NULL);
  assert(w->decoder_net_18_aligned_paddings_1_pad != NULL);
  assert(w->decoder_net_18_aligned_branches_0_net_1_cache_pad != NULL);
  assert(w->decoder_net_18_aligned_branches_0_net_1_weight != NULL);
  assert(w->decoder_net_18_aligned_branches_0_net_3_weight != NULL);
  assert(w->decoder_net_19_aligned_paddings_1_pad != NULL);
  assert(w->decoder_net_19_aligned_branches_0_net_1_cache_pad != NULL);
  assert(w->decoder_net_19_aligned_branches_0_net_1_weight != NULL);
  assert(w->decoder_net_19_aligned_branches_0_net_3_weight != NULL);
  assert(w->decoder_net_20_alpha != NULL);
  assert(w->decoder_noise_module_net_0_cache_pad != NULL);
  assert(w->decoder_noise_module_net_0_weight != NULL);
  assert(w->decoder_noise_module_net_1_alpha != NULL);
  assert(w->decoder_noise_module_net_2_cache_pad != NULL);
  assert(w->decoder_noise_module_net_2_weight != NULL);
  assert(w->decoder_noise_module_net_3_alpha != NULL);
  assert(w->decoder_noise_module_net_4_cache_pad != NULL);
  assert(w->decoder_noise_module_net_4_weight != NULL);
  assert(w->decoder_waveform_module_weight != NULL);
  assert(w->pqmf_inverse_conv_cache_pad != NULL);
  assert(w->pqmf_inverse_conv_weight != NULL);
  assert(w->ir_noise != NULL);

  w->skip = tensor_create(arena, U32_TPL(1), 8 * 2048);
  w->zeros = tensor_create(arena, U32_TPL(1, 16, 16, 5, 1), TENSOR_AUTO_CAP);
  w->mask = tensor_create(arena, U32_TPL(1, 16, 128), TENSOR_AUTO_CAP);
  w->hann = tensor_create(arena, U32_TPL(8), TENSOR_AUTO_CAP);
  w->scratch_1 = tensor_create(arena, U32_TPL(1), 2 * 8192);
  w->scratch_2 = tensor_create(arena, U32_TPL(1), 16 * 8192);

  assert(w->skip != NULL);
  assert(w->zeros != NULL);
  assert(w->mask!= NULL);
  assert(w->hann != NULL);
  assert(w->scratch_1 != NULL);
  assert(w->scratch_2 != NULL);

  tensor_hann(w->hann);
  tensor_fill(w->mask, 1.f);

  size_t channels = w->mask->dims[1];
  size_t len = w->mask->dims[2];
  
  for (size_t i = 1; i < channels; i += 2) {
    for (size_t j = 0; j < len; j += 2) {
      size_t idx = i * len + j; 
      w->mask->data[idx] = -1; 
    }
  }

  tensor_unsqueeze(w->latent_pca, w->latent_pca->rank);
  tensor_transpose(w->latent_pca, 0, 1);
  tensor_unsqueeze(w->latent_mean, 0);
  tensor_unsqueeze(w->latent_mean, w->latent_mean->rank);
}

void v2_load_weights_(arena_t* arena, v2_model_t* w) {
  w->noise                                              = tensor_load_from_file(arena, V2_BIN_PATH"pre_process_latent_noise.bin", TENSOR_AUTO_CAP);
  w->latent_pca                                         = tensor_load_from_file(arena, V2_BIN_PATH"latent_pca.bin", TENSOR_AUTO_CAP);
  w->latent_mean                                        = tensor_load_from_file(arena, V2_BIN_PATH"latent_mean.bin", TENSOR_AUTO_CAP);

  w->decoder_net_0_weight                               = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.0.weight.bin", TENSOR_AUTO_CAP);
  w->decoder_net_0_cache_pad                            = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.0.cache.pad.bin", TENSOR_AUTO_CAP);

  w->decoder_net_1_alpha                                = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.1.alpha.bin", TENSOR_AUTO_CAP);

  w->decoder_net_2_cache                                = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.2.cache.bin", TENSOR_AUTO_CAP);
  w->decoder_net_2_weight                               = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.2.weight.bin", TENSOR_AUTO_CAP);

  w->decoder_net_3_aligned_paddings_1_pad               = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.3.aligned.paddings.1.pad.bin", TENSOR_AUTO_CAP);
  w->decoder_net_3_aligned_branches_0_net_1_cache_pad   = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.3.aligned.branches.0.net.1.cache.pad.bin", TENSOR_AUTO_CAP);
  w->decoder_net_3_aligned_branches_0_net_1_weight      = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.3.aligned.branches.0.net.1.weight.bin", TENSOR_AUTO_CAP);
  w->decoder_net_3_aligned_branches_0_net_3_weight      = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.3.aligned.branches.0.net.3.weight.bin", TENSOR_AUTO_CAP);

  w->decoder_net_4_aligned_paddings_1_pad               = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.4.aligned.paddings.1.pad.bin", TENSOR_AUTO_CAP);
  w->decoder_net_4_aligned_branches_0_net_1_cache_pad   = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.4.aligned.branches.0.net.1.cache.pad.bin", TENSOR_AUTO_CAP);
  w->decoder_net_4_aligned_branches_0_net_1_weight      = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.4.aligned.branches.0.net.1.weight.bin", TENSOR_AUTO_CAP);
  w->decoder_net_4_aligned_branches_0_net_3_weight      = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.4.aligned.branches.0.net.3.weight.bin", TENSOR_AUTO_CAP);

  w->decoder_net_5_alpha                                = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.5.alpha.bin", TENSOR_AUTO_CAP);

  w->decoder_net_6_cache                                = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.6.cache.bin", TENSOR_AUTO_CAP);
  w->decoder_net_6_weight                               = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.6.weight.bin", TENSOR_AUTO_CAP);

  w->decoder_net_7_aligned_paddings_1_pad               = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.7.aligned.paddings.1.pad.bin", TENSOR_AUTO_CAP);
  w->decoder_net_7_aligned_branches_0_net_1_cache_pad   = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.7.aligned.branches.0.net.1.cache.pad.bin", TENSOR_AUTO_CAP);
  w->decoder_net_7_aligned_branches_0_net_1_weight      = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.7.aligned.branches.0.net.1.weight.bin", TENSOR_AUTO_CAP);
  w->decoder_net_7_aligned_branches_0_net_3_weight      = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.7.aligned.branches.0.net.3.weight.bin", TENSOR_AUTO_CAP);

  w->decoder_net_8_aligned_paddings_1_pad               = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.8.aligned.paddings.1.pad.bin", TENSOR_AUTO_CAP);
  w->decoder_net_8_aligned_branches_0_net_1_cache_pad   = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.8.aligned.branches.0.net.1.cache.pad.bin", TENSOR_AUTO_CAP);
  w->decoder_net_8_aligned_branches_0_net_1_weight      = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.8.aligned.branches.0.net.1.weight.bin", TENSOR_AUTO_CAP);
  w->decoder_net_8_aligned_branches_0_net_3_weight      = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.8.aligned.branches.0.net.3.weight.bin", TENSOR_AUTO_CAP);

  w->decoder_net_9_aligned_paddings_1_pad               = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.9.aligned.paddings.1.pad.bin", TENSOR_AUTO_CAP);
  w->decoder_net_9_aligned_branches_0_net_1_cache_pad   = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.9.aligned.branches.0.net.1.cache.pad.bin", TENSOR_AUTO_CAP);
  w->decoder_net_9_aligned_branches_0_net_1_weight      = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.9.aligned.branches.0.net.1.weight.bin", TENSOR_AUTO_CAP);
  w->decoder_net_9_aligned_branches_0_net_3_weight      = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.9.aligned.branches.0.net.3.weight.bin", TENSOR_AUTO_CAP);

  w->decoder_net_10_alpha                               = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.10.alpha.bin", TENSOR_AUTO_CAP);

  w->decoder_net_11_cache                                = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.11.cache.bin", TENSOR_AUTO_CAP);
  w->decoder_net_11_weight                              = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.11.weight.bin", TENSOR_AUTO_CAP);

  w->decoder_net_12_aligned_paddings_1_pad               = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.12.aligned.paddings.1.pad.bin", TENSOR_AUTO_CAP);
  w->decoder_net_12_aligned_branches_0_net_1_cache_pad   = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.12.aligned.branches.0.net.1.cache.pad.bin", TENSOR_AUTO_CAP);
  w->decoder_net_12_aligned_branches_0_net_1_weight      = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.12.aligned.branches.0.net.1.weight.bin", TENSOR_AUTO_CAP);
  w->decoder_net_12_aligned_branches_0_net_3_weight      = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.12.aligned.branches.0.net.3.weight.bin", TENSOR_AUTO_CAP);

  w->decoder_net_13_aligned_paddings_1_pad               = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.13.aligned.paddings.1.pad.bin", TENSOR_AUTO_CAP);
  w->decoder_net_13_aligned_branches_0_net_1_cache_pad   = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.13.aligned.branches.0.net.1.cache.pad.bin", TENSOR_AUTO_CAP);
  w->decoder_net_13_aligned_branches_0_net_1_weight      = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.13.aligned.branches.0.net.1.weight.bin", TENSOR_AUTO_CAP);
  w->decoder_net_13_aligned_branches_0_net_3_weight      = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.13.aligned.branches.0.net.3.weight.bin", TENSOR_AUTO_CAP);

  w->decoder_net_14_aligned_paddings_1_pad               = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.14.aligned.paddings.1.pad.bin", TENSOR_AUTO_CAP);
  w->decoder_net_14_aligned_branches_0_net_1_cache_pad   = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.14.aligned.branches.0.net.1.cache.pad.bin", TENSOR_AUTO_CAP);
  w->decoder_net_14_aligned_branches_0_net_1_weight      = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.14.aligned.branches.0.net.1.weight.bin", TENSOR_AUTO_CAP);
  w->decoder_net_14_aligned_branches_0_net_3_weight      = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.14.aligned.branches.0.net.3.weight.bin", TENSOR_AUTO_CAP);

  w->decoder_net_15_alpha                               = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.15.alpha.bin", TENSOR_AUTO_CAP);

  w->decoder_net_16_cache                                = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.16.cache.bin", TENSOR_AUTO_CAP);
  w->decoder_net_16_weight                              = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.16.weight.bin", TENSOR_AUTO_CAP);

  w->decoder_net_17_aligned_paddings_1_pad               = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.17.aligned.paddings.1.pad.bin", TENSOR_AUTO_CAP);
  w->decoder_net_17_aligned_branches_0_net_1_cache_pad   = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.17.aligned.branches.0.net.1.cache.pad.bin", TENSOR_AUTO_CAP);
  w->decoder_net_17_aligned_branches_0_net_1_weight      = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.17.aligned.branches.0.net.1.weight.bin", TENSOR_AUTO_CAP);
  w->decoder_net_17_aligned_branches_0_net_3_weight      = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.17.aligned.branches.0.net.3.weight.bin", TENSOR_AUTO_CAP);

  w->decoder_net_18_aligned_paddings_1_pad               = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.18.aligned.paddings.1.pad.bin", TENSOR_AUTO_CAP);
  w->decoder_net_18_aligned_branches_0_net_1_cache_pad   = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.18.aligned.branches.0.net.1.cache.pad.bin", TENSOR_AUTO_CAP);
  w->decoder_net_18_aligned_branches_0_net_1_weight      = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.18.aligned.branches.0.net.1.weight.bin", TENSOR_AUTO_CAP);
  w->decoder_net_18_aligned_branches_0_net_3_weight      = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.18.aligned.branches.0.net.3.weight.bin", TENSOR_AUTO_CAP);

  w->decoder_net_19_aligned_paddings_1_pad               = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.19.aligned.paddings.1.pad.bin", TENSOR_AUTO_CAP);
  w->decoder_net_19_aligned_branches_0_net_1_cache_pad   = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.19.aligned.branches.0.net.1.cache.pad.bin", TENSOR_AUTO_CAP);
  w->decoder_net_19_aligned_branches_0_net_1_weight      = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.19.aligned.branches.0.net.1.weight.bin", TENSOR_AUTO_CAP);
  w->decoder_net_19_aligned_branches_0_net_3_weight      = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.19.aligned.branches.0.net.3.weight.bin", TENSOR_AUTO_CAP);

  w->decoder_net_20_alpha                               = tensor_load_from_file(arena, V2_BIN_PATH"decoder.net.20.alpha.bin", TENSOR_AUTO_CAP);

  w->decoder_noise_module_net_0_cache_pad               = tensor_load_from_file(arena, V2_BIN_PATH"decoder.noise_module.net.0.cache.pad.bin", TENSOR_AUTO_CAP);
  w->decoder_noise_module_net_0_weight                  = tensor_load_from_file(arena, V2_BIN_PATH"decoder.noise_module.net.0.weight.bin", TENSOR_AUTO_CAP);
  w->decoder_noise_module_net_1_alpha                   = tensor_load_from_file(arena, V2_BIN_PATH"decoder.noise_module.net.1.alpha.bin", TENSOR_AUTO_CAP);
  w->decoder_noise_module_net_2_cache_pad               = tensor_load_from_file(arena, V2_BIN_PATH"decoder.noise_module.net.2.cache.pad.bin", TENSOR_AUTO_CAP);
  w->decoder_noise_module_net_2_weight                  = tensor_load_from_file(arena, V2_BIN_PATH"decoder.noise_module.net.2.weight.bin", TENSOR_AUTO_CAP);
  w->decoder_noise_module_net_3_alpha                   = tensor_load_from_file(arena, V2_BIN_PATH"decoder.noise_module.net.3.alpha.bin", TENSOR_AUTO_CAP);
  w->decoder_noise_module_net_4_cache_pad               = tensor_load_from_file(arena, V2_BIN_PATH"decoder.noise_module.net.4.cache.pad.bin", TENSOR_AUTO_CAP);
  w->decoder_noise_module_net_4_weight                  = tensor_load_from_file(arena, V2_BIN_PATH"decoder.noise_module.net.4.weight.bin", TENSOR_AUTO_CAP);

  w->decoder_waveform_module_cache_pad                  = tensor_load_from_file(arena, V2_BIN_PATH"decoder.waveform_module.cache.pad.bin", TENSOR_AUTO_CAP);
  w->decoder_waveform_module_weight                     = tensor_load_from_file(arena, V2_BIN_PATH"decoder.waveform_module.weight.bin", TENSOR_AUTO_CAP);
  w->pqmf_inverse_conv_cache_pad                        = tensor_load_from_file(arena, V2_BIN_PATH"pqmf.inverse_conv.cache.pad.bin", TENSOR_AUTO_CAP);
  w->pqmf_inverse_conv_weight                           = tensor_load_from_file(arena, V2_BIN_PATH"pqmf.inverse_conv.weight.bin", TENSOR_AUTO_CAP);
  w->ir_noise                                           = tensor_load_from_file(arena, V2_BIN_PATH"ir_noise.bin", 12 * 8192);

  assert(w->noise != NULL);
  assert(w->latent_pca != NULL);
  assert(w->latent_mean != NULL);

  assert(w->decoder_net_0_weight != NULL);
  assert(w->decoder_net_0_cache_pad != NULL);

  assert(w->decoder_net_1_alpha != NULL);

  assert(w->decoder_net_2_cache != NULL);
  assert(w->decoder_net_2_weight != NULL);

  assert(w->decoder_net_3_aligned_paddings_1_pad != NULL);
  assert(w->decoder_net_3_aligned_branches_0_net_1_cache_pad != NULL);
  assert(w->decoder_net_3_aligned_branches_0_net_1_weight != NULL);
  assert(w->decoder_net_3_aligned_branches_0_net_3_weight != NULL);

  assert(w->decoder_net_4_aligned_paddings_1_pad != NULL);
  assert(w->decoder_net_4_aligned_branches_0_net_1_cache_pad != NULL);
  assert(w->decoder_net_4_aligned_branches_0_net_1_weight != NULL);
  assert(w->decoder_net_4_aligned_branches_0_net_3_weight != NULL);

  assert(w->decoder_net_5_alpha != NULL);

  assert(w->decoder_net_6_cache != NULL);
  assert(w->decoder_net_6_weight != NULL);

  assert(w->decoder_net_7_aligned_paddings_1_pad != NULL);
  assert(w->decoder_net_7_aligned_branches_0_net_1_cache_pad != NULL);
  assert(w->decoder_net_7_aligned_branches_0_net_1_weight != NULL);
  assert(w->decoder_net_7_aligned_branches_0_net_3_weight != NULL);

  assert(w->decoder_net_8_aligned_paddings_1_pad != NULL);
  assert(w->decoder_net_8_aligned_branches_0_net_1_cache_pad != NULL);
  assert(w->decoder_net_8_aligned_branches_0_net_1_weight != NULL);
  assert(w->decoder_net_8_aligned_branches_0_net_3_weight != NULL);

  assert(w->decoder_net_9_aligned_paddings_1_pad != NULL);
  assert(w->decoder_net_9_aligned_branches_0_net_1_cache_pad != NULL);
  assert(w->decoder_net_9_aligned_branches_0_net_1_weight != NULL);
  assert(w->decoder_net_9_aligned_branches_0_net_3_weight != NULL);

  assert(w->decoder_net_10_alpha != NULL);

  assert(w->decoder_net_11_cache != NULL);
  assert(w->decoder_net_11_weight != NULL);

  assert(w->decoder_net_12_aligned_paddings_1_pad != NULL);
  assert(w->decoder_net_12_aligned_branches_0_net_1_cache_pad != NULL);
  assert(w->decoder_net_12_aligned_branches_0_net_1_weight != NULL);
  assert(w->decoder_net_12_aligned_branches_0_net_3_weight != NULL);

  assert(w->decoder_net_13_aligned_paddings_1_pad != NULL);
  assert(w->decoder_net_13_aligned_branches_0_net_1_cache_pad != NULL);
  assert(w->decoder_net_13_aligned_branches_0_net_1_weight != NULL);
  assert(w->decoder_net_13_aligned_branches_0_net_3_weight != NULL);

  assert(w->decoder_net_14_aligned_paddings_1_pad != NULL);
  assert(w->decoder_net_14_aligned_branches_0_net_1_cache_pad != NULL);
  assert(w->decoder_net_14_aligned_branches_0_net_1_weight != NULL);
  assert(w->decoder_net_14_aligned_branches_0_net_3_weight != NULL);

  assert(w->decoder_net_15_alpha != NULL);

  assert(w->decoder_net_16_cache != NULL);
  assert(w->decoder_net_16_weight != NULL);

  assert(w->decoder_net_17_aligned_paddings_1_pad != NULL);
  assert(w->decoder_net_17_aligned_branches_0_net_1_cache_pad != NULL);
  assert(w->decoder_net_17_aligned_branches_0_net_1_weight != NULL);
  assert(w->decoder_net_17_aligned_branches_0_net_3_weight != NULL);

  assert(w->decoder_net_18_aligned_paddings_1_pad != NULL);
  assert(w->decoder_net_18_aligned_branches_0_net_1_cache_pad != NULL);
  assert(w->decoder_net_18_aligned_branches_0_net_1_weight != NULL);
  assert(w->decoder_net_18_aligned_branches_0_net_3_weight != NULL);

  assert(w->decoder_net_19_aligned_paddings_1_pad != NULL);
  assert(w->decoder_net_19_aligned_branches_0_net_1_cache_pad != NULL);
  assert(w->decoder_net_19_aligned_branches_0_net_1_weight != NULL);
  assert(w->decoder_net_19_aligned_branches_0_net_3_weight != NULL);

  assert(w->decoder_net_20_alpha != NULL);

  assert(w->decoder_noise_module_net_0_cache_pad != NULL);
  assert(w->decoder_noise_module_net_0_weight != NULL);
  assert(w->decoder_noise_module_net_1_alpha != NULL);
  assert(w->decoder_noise_module_net_2_cache_pad != NULL);
  assert(w->decoder_noise_module_net_2_weight != NULL);
  assert(w->decoder_noise_module_net_3_alpha != NULL);
  assert(w->decoder_noise_module_net_4_cache_pad != NULL);
  assert(w->decoder_noise_module_net_4_weight != NULL);
  assert(w->decoder_waveform_module_weight != NULL);
  assert(w->pqmf_inverse_conv_cache_pad != NULL);
  assert(w->pqmf_inverse_conv_weight != NULL);
  assert(w->ir_noise != NULL);

  w->skip = tensor_create(arena, U32_TPL(1), 8 * 2048);
  w->zeros = tensor_create(arena, U32_TPL(1, 16, 16, 5, 1), TENSOR_AUTO_CAP);
  w->mask = tensor_create(arena, U32_TPL(1, 16, 128), TENSOR_AUTO_CAP);
  w->hann = tensor_create(arena, U32_TPL(8), TENSOR_AUTO_CAP);
  w->scratch_1 = tensor_create(arena, U32_TPL(1), 2 * 8192);
  w->scratch_2 = tensor_create(arena, U32_TPL(1), 16 * 8192);

  assert(w->skip != NULL);
  assert(w->zeros != NULL);
  assert(w->mask!= NULL);
  assert(w->hann != NULL);
  assert(w->scratch_1 != NULL);
  assert(w->scratch_2 != NULL);

  tensor_hann(w->hann);
  tensor_fill(w->mask, 1.f);

  size_t channels = w->mask->dims[1];
  size_t len = w->mask->dims[2];
  
  for (size_t i = 1; i < channels; i += 2) {
    for (size_t j = 0; j < len; j += 2) {
      size_t idx = i * len + j; 
      w->mask->data[idx] = -1; 
    }
  }

  tensor_unsqueeze(w->latent_pca, w->latent_pca->rank);
  tensor_transpose(w->latent_pca, 0, 1);
  tensor_unsqueeze(w->latent_mean, 0);
  tensor_unsqueeze(w->latent_mean, w->latent_mean->rank);
}

void v2_noise_generator(tensor_t* input, v2_model_t* w) {
  // TODO(luca): verify that we actually need this amp stuff as it seems to not
  // really be contributing anything.

  v2_cached_conv1d(input, w->decoder_noise_module_net_0_weight, w->decoder_noise_module_net_0_cache_pad, 2, 1);

  tensor_snake(input, w->decoder_noise_module_net_1_alpha);
  v2_cached_conv1d(input, w->decoder_noise_module_net_2_weight, w->decoder_noise_module_net_2_cache_pad, 2, 1);
  tensor_snake(input, w->decoder_noise_module_net_3_alpha);
  v2_cached_conv1d(input, w->decoder_noise_module_net_4_weight, w->decoder_noise_module_net_4_cache_pad, 2, 1);
  tensor_add(input, -5.f);

  tensor_sigmoid(input);
  tensor_pow(input, 2.3f);
  tensor_mul(input, 2.f);
  tensor_add(input, 1e-7f);
  tensor_permute(input, U32_TPL(0, 2, 1));
  tensor_reshape(input, U32_TPL(1, 16, 16, 5));
  tensor_unsqueeze(input, input->rank);
  tensor_cat(input, w->zeros, input->rank - 1, CRV_BACK);
  tensor_irfft(input);
  tensor_roll(input, 4, input->rank - 1);
  tensor_tmul_last_dim(input, w->hann);
  tensor_roll(input, -4, input->rank - 1);

  tensor_t* ir_noise = w->scratch_2;
  tensor_copy(ir_noise, w->ir_noise);
  tensor_pad(ir_noise, 0, 8);
  tensor_pad(input, 8, 0);

  tensor_rfft(ir_noise);
  tensor_rfft(input);

  tensor_tmul_last_dim(input, ir_noise);
  tensor_irfft(input);
   
  {
    size_t final_count = input->count / 2;
    float* x = input->data;
    float* y = input->swap;

    size_t i = 8;
    size_t j = 0;
    while (j < final_count) {
      y[j++] = x[i++];
      y[j++] = x[i++];
      y[j++] = x[i++];
      y[j++] = x[i++];
      y[j++] = x[i++];
      y[j++] = x[i++];
      y[j++] = x[i++];
      y[j++] = x[i++];
      i += 8;
    }

    input->data = y;
    input->swap = x;
    input->dims[input->rank - 1] /= 2;
    input->count = final_count;
  }

  tensor_permute(input, U32_TPL(0, 2, 1, 3));
  tensor_reshape(input, U32_TPL(1, 16, 128));
}

void v2_block(tensor_t* input, tensor_t* skip, tensor_t* w0, tensor_t* w1, tensor_t* c0, tensor_t* c1, size_t dilation) {
  tensor_copy(skip, input);
  v2_cached_pad(skip, c0);
  tensor_leaky_relu(input, 0.2);
  v2_cached_conv1d(input, w0, c1, 1, dilation);
  tensor_leaky_relu(input, 0.2);
  tensor_conv1d(input, w1, 1, 1);
  tensor_tadd(input, skip);
}

void v2_decode(tensor_t* z, v2_model_t* w) {
  tensor_cat(z, w->noise, 1, CRV_BACK);
  tensor_conv1d(z, w->latent_pca, 1, 1);
  tensor_tadd(z, w->latent_mean);

  v2_cached_conv1d(z, w->decoder_net_0_weight, w->decoder_net_0_cache_pad, 1, 1);
  tensor_snake(z, w->decoder_net_1_alpha);
  v2_cached_conv_transpose1d(z, w->decoder_net_2_weight, w->decoder_net_2_cache, 2, 1);

  v2_block(
    z,
    w->skip,
    w->decoder_net_3_aligned_branches_0_net_1_weight,
    w->decoder_net_3_aligned_branches_0_net_3_weight,
    w->decoder_net_3_aligned_paddings_1_pad,
    w->decoder_net_3_aligned_branches_0_net_1_cache_pad,
    1
  );

  v2_block(
    z,
    w->skip,
    w->decoder_net_4_aligned_branches_0_net_1_weight,
    w->decoder_net_4_aligned_branches_0_net_3_weight,
    w->decoder_net_4_aligned_paddings_1_pad,
    w->decoder_net_4_aligned_branches_0_net_1_cache_pad,
    3
  );

  tensor_snake(z, w->decoder_net_5_alpha);
  v2_cached_conv_transpose1d(z, w->decoder_net_6_weight, w->decoder_net_6_cache, 4, 1);

  v2_block(
    z,
    w->skip,
    w->decoder_net_7_aligned_branches_0_net_1_weight,
    w->decoder_net_7_aligned_branches_0_net_3_weight,
    w->decoder_net_7_aligned_paddings_1_pad,
    w->decoder_net_7_aligned_branches_0_net_1_cache_pad,
    1
  );

  v2_block(
    z,
    w->skip,
    w->decoder_net_8_aligned_branches_0_net_1_weight,
    w->decoder_net_8_aligned_branches_0_net_3_weight,
    w->decoder_net_8_aligned_paddings_1_pad,
    w->decoder_net_8_aligned_branches_0_net_1_cache_pad,
    3
  );

  v2_block(
    z,
    w->skip,
    w->decoder_net_9_aligned_branches_0_net_1_weight,
    w->decoder_net_9_aligned_branches_0_net_3_weight,
    w->decoder_net_9_aligned_paddings_1_pad,
    w->decoder_net_9_aligned_branches_0_net_1_cache_pad,
    9
  );

  tensor_snake(z, w->decoder_net_10_alpha);
  v2_cached_conv_transpose1d(z, w->decoder_net_11_weight, w->decoder_net_11_cache, 4, 1);

  v2_block(
    z,
    w->skip,
    w->decoder_net_12_aligned_branches_0_net_1_weight,
    w->decoder_net_12_aligned_branches_0_net_3_weight,
    w->decoder_net_12_aligned_paddings_1_pad,
    w->decoder_net_12_aligned_branches_0_net_1_cache_pad,
    1
  );

  v2_block(
    z,
    w->skip,
    w->decoder_net_13_aligned_branches_0_net_1_weight,
    w->decoder_net_13_aligned_branches_0_net_3_weight,
    w->decoder_net_13_aligned_paddings_1_pad,
    w->decoder_net_13_aligned_branches_0_net_1_cache_pad,
    3
  );

  v2_block(
    z,
    w->skip,
    w->decoder_net_14_aligned_branches_0_net_1_weight,
    w->decoder_net_14_aligned_branches_0_net_3_weight,
    w->decoder_net_14_aligned_paddings_1_pad,
    w->decoder_net_14_aligned_branches_0_net_1_cache_pad,
    9
  );

  tensor_snake(z, w->decoder_net_15_alpha);
  v2_cached_conv_transpose1d(z, w->decoder_net_16_weight, w->decoder_net_16_cache, 4, 1);

  v2_block(
    z,
    w->skip,
    w->decoder_net_17_aligned_branches_0_net_1_weight,
    w->decoder_net_17_aligned_branches_0_net_3_weight,
    w->decoder_net_17_aligned_paddings_1_pad,
    w->decoder_net_17_aligned_branches_0_net_1_cache_pad,
    1
  );

  v2_block(
    z,
    w->skip,
    w->decoder_net_18_aligned_branches_0_net_1_weight,
    w->decoder_net_18_aligned_branches_0_net_3_weight,
    w->decoder_net_18_aligned_paddings_1_pad,
    w->decoder_net_18_aligned_branches_0_net_1_cache_pad,
    3
  );

  v2_block(
    z,
    w->skip,
    w->decoder_net_19_aligned_branches_0_net_1_weight,
    w->decoder_net_19_aligned_branches_0_net_3_weight,
    w->decoder_net_19_aligned_paddings_1_pad,
    w->decoder_net_19_aligned_branches_0_net_1_cache_pad,
    9
  );

  tensor_snake(z, w->decoder_net_20_alpha);

  tensor_t* noise = w->scratch_1;
  tensor_copy(noise, z);

  v2_noise_generator(noise, w);

  // Waveform module
  v2_cached_conv1d(z, w->decoder_waveform_module_weight, w->decoder_waveform_module_cache_pad, 1, 1);

  // Amplitude modulation
  tensor_t* amp = w->scratch_2;
  tensor_split(amp, z);
  tensor_sigmoid(amp);
  tensor_tmul(z, amp);
  tensor_tadd(z, noise); 
  tensor_tanh(z);

  // Post
  tensor_reshape(z, U32_TPL(1, 16, 128)); 
  tensor_tmul(z, w->mask);
  v2_cached_conv1d(z, w->pqmf_inverse_conv_weight, w->pqmf_inverse_conv_cache_pad, 1, 1);
  tensor_mul(z, 16);
  tensor_flip(z, 1);
  tensor_permute(z, U32_TPL(0, 2, 1));
  tensor_reshape(z, U32_TPL(1, 128, 1, 16));
  tensor_permute(z, U32_TPL(0, 2, 1, 3));
  tensor_reshape(z, U32_TPL(1, 1, 2048));
}

#endif // V2_H
