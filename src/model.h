#ifndef MODEL_H
#define MODEL_H
#define MODEL_LOAD_ERROR 0
#define MODEL_LOAD_SUCCESS 1
#define CRV_IMPLEMENTATION
#include "crave.h"

typedef struct {
  uint64_t size;
  uint64_t config;
  uint32_t block_size;
  uint32_t num_latents;
  uint32_t sample_rate;
  uint32_t num_tensors;
} header_t;

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
} v1_model_t;

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

typedef struct {
  header_t header;

  union {
    v1_model_t v1;
    v2_model_t v2;
  };
} model_t;

int model_load_file_into_memory(void* dest, char* filepath, size_t* size) {
  int result = MODEL_LOAD_ERROR;

  assert(dest);
  assert(filepath);
  assert(size);

  FILE* file = fopen(filepath, "rb"); 
  if (file) {
    fseek(file, 0, SEEK_END);
    *size = ftell(file);
    rewind(file);
    if (*size) {
      if (fread(dest, 1, *size, file) == *size) {
        return MODEL_LOAD_SUCCESS;    
      } else {
        fprintf(stderr, "Failed to read entire file.\n");
      }
    } else {
      fprintf(stderr, "Model file is empty.\n");
    }
  } else {
    fprintf(stderr, "Failed to open file.\n");
  }

  fclose(file);

  return result;
}

void model_get_header_from_memory(header_t* dest, void* src) {
  memcpy(dest, src, sizeof(header_t));
}

int v1_load(char** dest, v1_model_t* w, tensor_list_t* list) {
  w->noise                                            = crv_tensor_find_in_list(list, "pre_process_latent_noise");
  w->latent_pca                                       = crv_tensor_find_in_list(list, "pre_process_latent_latent_pca");
  w->latent_mean                                      = crv_tensor_find_in_list(list, "pre_process_latent_latent_mean");
  w->pqmf_inverse_conv_weight                         = crv_tensor_find_in_list(list, "pqmf.inverse_conv.weight");
  w->net_0_weight                                     = crv_tensor_find_in_list(list, "decoder.net.0.weight");
  w->net_1_alpha                                      = crv_tensor_find_in_list(list, "decoder.net.1.alpha");
  w->net_2_weight                                     = crv_tensor_find_in_list(list, "decoder.net.2.weight");
  w->decoder_net_3_aligned_branches_0_net_1_weight    = crv_tensor_find_in_list(list, "decoder.net.3.aligned.branches.0.net.1.weight");
  w->decoder_net_3_aligned_branches_0_net_3_weight    = crv_tensor_find_in_list(list, "decoder.net.3.aligned.branches.0.net.3.weight");
  w->decoder_net_4_aligned_branches_0_net_1_weight    = crv_tensor_find_in_list(list, "decoder.net.4.aligned.branches.0.net.1.weight");
  w->decoder_net_4_aligned_branches_0_net_3_weight    = crv_tensor_find_in_list(list, "decoder.net.4.aligned.branches.0.net.3.weight");
  w->net_5_alpha                                      = crv_tensor_find_in_list(list, "decoder.net.5.alpha");
  w->net_6_weight                                     = crv_tensor_find_in_list(list, "decoder.net.6.weight");
  w->decoder_net_7_aligned_branches_0_net_1_weight    = crv_tensor_find_in_list(list, "decoder.net.7.aligned.branches.0.net.1.weight");
  w->decoder_net_7_aligned_branches_0_net_3_weight    = crv_tensor_find_in_list(list, "decoder.net.7.aligned.branches.0.net.3.weight");
  w->decoder_net_8_aligned_branches_0_net_1_weight    = crv_tensor_find_in_list(list, "decoder.net.8.aligned.branches.0.net.1.weight");
  w->decoder_net_8_aligned_branches_0_net_3_weight    = crv_tensor_find_in_list(list, "decoder.net.8.aligned.branches.0.net.3.weight");
  w->decoder_net_9_aligned_branches_0_net_1_weight    = crv_tensor_find_in_list(list, "decoder.net.9.aligned.branches.0.net.1.weight");
  w->decoder_net_9_aligned_branches_0_net_3_weight    = crv_tensor_find_in_list(list, "decoder.net.9.aligned.branches.0.net.3.weight");
  w->net_10_alpha                                     = crv_tensor_find_in_list(list, "decoder.net.10.alpha");
  w->net_11_weight                                    = crv_tensor_find_in_list(list, "decoder.net.11.weight");
  w->decoder_net_12_aligned_branches_0_net_1_weight   = crv_tensor_find_in_list(list, "decoder.net.12.aligned.branches.0.net.1.weight");
  w->decoder_net_12_aligned_branches_0_net_3_weight   = crv_tensor_find_in_list(list, "decoder.net.12.aligned.branches.0.net.3.weight");
  w->decoder_net_13_aligned_branches_0_net_1_weight   = crv_tensor_find_in_list(list, "decoder.net.13.aligned.branches.0.net.1.weight");
  w->decoder_net_13_aligned_branches_0_net_3_weight   = crv_tensor_find_in_list(list, "decoder.net.13.aligned.branches.0.net.3.weight");
  w->decoder_net_14_aligned_branches_0_net_1_weight   = crv_tensor_find_in_list(list, "decoder.net.14.aligned.branches.0.net.1.weight");
  w->decoder_net_14_aligned_branches_0_net_3_weight   = crv_tensor_find_in_list(list, "decoder.net.14.aligned.branches.0.net.3.weight");
  w->net_15_alpha                                     = crv_tensor_find_in_list(list, "decoder.net.15.alpha");
  w->net_16_weight                                    = crv_tensor_find_in_list(list, "decoder.net.16.weight");
  w->decoder_net_17_aligned_branches_0_net_1_weight   = crv_tensor_find_in_list(list, "decoder.net.17.aligned.branches.0.net.1.weight");
  w->decoder_net_17_aligned_branches_0_net_3_weight   = crv_tensor_find_in_list(list, "decoder.net.17.aligned.branches.0.net.3.weight");
  w->decoder_net_18_aligned_branches_0_net_1_weight   = crv_tensor_find_in_list(list, "decoder.net.18.aligned.branches.0.net.1.weight");
  w->decoder_net_18_aligned_branches_0_net_3_weight   = crv_tensor_find_in_list(list, "decoder.net.18.aligned.branches.0.net.3.weight");
  w->decoder_net_19_aligned_branches_0_net_1_weight   = crv_tensor_find_in_list(list, "decoder.net.19.aligned.branches.0.net.1.weight");
  w->decoder_net_19_aligned_branches_0_net_3_weight   = crv_tensor_find_in_list(list, "decoder.net.19.aligned.branches.0.net.3.weight");
  w->net_20_alpha                                     = crv_tensor_find_in_list(list, "decoder.net.20.alpha");
  w->net_21_weight                                    = crv_tensor_find_in_list(list, "decoder.net.21.weight");

  if (w->noise == NULL)                                             return MODEL_LOAD_ERROR;
  if (w->latent_pca == NULL)                                        return MODEL_LOAD_ERROR;
  if (w->latent_mean == NULL)                                       return MODEL_LOAD_ERROR;
  if (w->pqmf_inverse_conv_weight == NULL)                          return MODEL_LOAD_ERROR;
  if (w->net_0_weight == NULL)                                      return MODEL_LOAD_ERROR;
  if (w->net_1_alpha == NULL)                                       return MODEL_LOAD_ERROR;
  if (w->net_2_weight == NULL)                                      return MODEL_LOAD_ERROR;
  if (w->decoder_net_3_aligned_branches_0_net_1_weight == NULL)     return MODEL_LOAD_ERROR;
  if (w->decoder_net_3_aligned_branches_0_net_3_weight == NULL)     return MODEL_LOAD_ERROR;
  if (w->decoder_net_4_aligned_branches_0_net_1_weight == NULL)     return MODEL_LOAD_ERROR;
  if (w->decoder_net_4_aligned_branches_0_net_3_weight == NULL)     return MODEL_LOAD_ERROR;
  if (w->net_5_alpha == NULL)                                       return MODEL_LOAD_ERROR;
  if (w->net_6_weight == NULL)                                      return MODEL_LOAD_ERROR;
  if (w->decoder_net_7_aligned_branches_0_net_1_weight == NULL)     return MODEL_LOAD_ERROR;
  if (w->decoder_net_7_aligned_branches_0_net_3_weight == NULL)     return MODEL_LOAD_ERROR;
  if (w->decoder_net_8_aligned_branches_0_net_1_weight == NULL)     return MODEL_LOAD_ERROR;
  if (w->decoder_net_8_aligned_branches_0_net_3_weight == NULL)     return MODEL_LOAD_ERROR;
  if (w->decoder_net_9_aligned_branches_0_net_1_weight == NULL)     return MODEL_LOAD_ERROR;
  if (w->decoder_net_9_aligned_branches_0_net_3_weight == NULL)     return MODEL_LOAD_ERROR;
  if (w->net_10_alpha == NULL)                                      return MODEL_LOAD_ERROR;
  if (w->net_11_weight == NULL)                                     return MODEL_LOAD_ERROR;
  if (w->decoder_net_12_aligned_branches_0_net_1_weight == NULL)    return MODEL_LOAD_ERROR;
  if (w->decoder_net_12_aligned_branches_0_net_3_weight == NULL)    return MODEL_LOAD_ERROR;
  if (w->decoder_net_13_aligned_branches_0_net_1_weight == NULL)    return MODEL_LOAD_ERROR;
  if (w->decoder_net_13_aligned_branches_0_net_3_weight == NULL)    return MODEL_LOAD_ERROR;
  if (w->decoder_net_14_aligned_branches_0_net_1_weight == NULL)    return MODEL_LOAD_ERROR;
  if (w->decoder_net_14_aligned_branches_0_net_3_weight == NULL)    return MODEL_LOAD_ERROR;
  if (w->net_15_alpha == NULL)                                      return MODEL_LOAD_ERROR;
  if (w->net_16_weight == NULL)                                     return MODEL_LOAD_ERROR;
  if (w->decoder_net_17_aligned_branches_0_net_1_weight == NULL)    return MODEL_LOAD_ERROR;
  if (w->decoder_net_17_aligned_branches_0_net_3_weight == NULL)    return MODEL_LOAD_ERROR;
  if (w->decoder_net_18_aligned_branches_0_net_1_weight == NULL)    return MODEL_LOAD_ERROR;
  if (w->decoder_net_18_aligned_branches_0_net_3_weight == NULL)    return MODEL_LOAD_ERROR;
  if (w->decoder_net_19_aligned_branches_0_net_1_weight == NULL)    return MODEL_LOAD_ERROR;
  if (w->decoder_net_19_aligned_branches_0_net_3_weight == NULL)    return MODEL_LOAD_ERROR;
  if (w->net_20_alpha == NULL)                                      return MODEL_LOAD_ERROR;
  if (w->net_21_weight == NULL)                                     return MODEL_LOAD_ERROR;

  w->skip = crv_tensor_create(dest, CRV_TPL(2, 1024), CRV_TENSOR_AUTO_CAP, CRV_NO_SWAP);
  w->mask = crv_tensor_create(dest, CRV_TPL(1, 16, 128), CRV_TENSOR_AUTO_CAP, CRV_NO_SWAP);

  crv_tensor_fill(w->mask, 1.f);

  size_t channels = w->mask->dims[1];
  size_t len = w->mask->dims[2];
  
  for (size_t i = 1; i < channels; i += 2) {
    for (size_t j = 0; j < len; j += 2) {
      size_t idx = i * len + j; 
      w->mask->data[idx] = -1; 
    }
  }

  w->latent_pca->swap = (float*)*dest;
  *dest += w->latent_pca->count * sizeof(float);

  crv_tensor_unsqueeze(w->latent_pca, w->latent_pca->rank);
  crv_tensor_transpose(w->latent_pca, 0, 1);
  crv_tensor_unsqueeze(w->latent_mean, 0);
  crv_tensor_unsqueeze(w->latent_mean, w->latent_mean->rank);

  return MODEL_LOAD_SUCCESS;
}

void v1_decode(tensor_t* z, v1_model_t* w) {
  crv_tensor_cat(z, w->noise, 1, CRV_BACK);
  crv_tensor_conv1d(z, w->latent_pca, 1, 1);
  crv_tensor_tadd(z, w->latent_mean);

  // (0)
  crv_tensor_pad(z, 2, 0);
  crv_tensor_conv1d(z, w->net_0_weight, 1, 1);
  crv_tensor_init(z, CRV_TPL(1, 256, 1));

  // (1)
  crv_tensor_snake(z, w->net_1_alpha);

  // (2)
  crv_tensor_pad(z, 1, 0);
  crv_tensor_conv_transpose1d(z, w->net_2_weight, 2, 1);
  crv_tensor_trunc(z, 2, 2);

  // (3)
  crv_tensor_copy(w->skip, z);
  crv_tensor_leaky_relu(z, 0.2f);
  crv_tensor_pad(z, 2, 0);
  crv_tensor_conv1d(z, w->decoder_net_3_aligned_branches_0_net_1_weight, 1, 1);
  crv_tensor_leaky_relu(z, 0.2f);
  crv_tensor_conv1d(z, w->decoder_net_3_aligned_branches_0_net_3_weight, 1, 1);
  crv_tensor_tadd(z, w->skip);

  // (4)
  crv_tensor_copy(w->skip, z);
  crv_tensor_leaky_relu(z, 0.2f);
  crv_tensor_pad(z, 6, 0);
  crv_tensor_conv1d(z, w->decoder_net_4_aligned_branches_0_net_1_weight, 1, 3);
  crv_tensor_leaky_relu(z, 0.2f);
  crv_tensor_conv1d(z, w->decoder_net_4_aligned_branches_0_net_3_weight, 1, 1);
  crv_tensor_tadd(z, w->skip);

  // (5)
  crv_tensor_snake(z, w->net_5_alpha);

  // (6)
  crv_tensor_pad(z, 1, 0);
  crv_tensor_conv_transpose1d(z, w->net_6_weight, 4, 1);
  crv_tensor_trunc(z, 4, 4);

  // (7)
  crv_tensor_copy(w->skip, z);
  crv_tensor_leaky_relu(z, 0.2f);
  crv_tensor_pad(z, 2, 0);
  crv_tensor_conv1d(z, w->decoder_net_7_aligned_branches_0_net_1_weight, 1, 1);
  crv_tensor_leaky_relu(z, 0.2f);
  crv_tensor_conv1d(z, w->decoder_net_7_aligned_branches_0_net_3_weight, 1, 1);
  crv_tensor_tadd(z, w->skip);

  // (8)
  crv_tensor_copy(w->skip, z);
  crv_tensor_leaky_relu(z, 0.2f);
  crv_tensor_pad(z, 6, 0);
  crv_tensor_conv1d(z, w->decoder_net_8_aligned_branches_0_net_1_weight, 1, 3);
  crv_tensor_leaky_relu(z, 0.2f);
  crv_tensor_conv1d(z, w->decoder_net_8_aligned_branches_0_net_3_weight, 1, 1);
  crv_tensor_tadd(z, w->skip);

  // (9)
  crv_tensor_copy(w->skip, z);
  crv_tensor_leaky_relu(z, 0.2f);
  crv_tensor_pad(z, 18, 0);
  crv_tensor_conv1d(z, w->decoder_net_9_aligned_branches_0_net_1_weight, 1, 9);
  crv_tensor_leaky_relu(z, 0.2f);
  crv_tensor_conv1d(z, w->decoder_net_9_aligned_branches_0_net_3_weight, 1, 1);
  crv_tensor_tadd(z, w->skip);

  // (10)
  crv_tensor_snake(z, w->net_10_alpha);

  // (11)
  crv_tensor_pad(z, 1, 0);
  crv_tensor_conv_transpose1d(z, w->net_11_weight, 4, 1);
  crv_tensor_trunc(z, 4, 4);

  // (12)
  crv_tensor_copy(w->skip, z);
  crv_tensor_leaky_relu(z, 0.2f);
  crv_tensor_pad(z, 2, 0);
  crv_tensor_conv1d(z, w->decoder_net_12_aligned_branches_0_net_1_weight, 1, 1);
  crv_tensor_leaky_relu(z, 0.2f);
  crv_tensor_conv1d(z, w->decoder_net_12_aligned_branches_0_net_3_weight, 1, 1);
  crv_tensor_tadd(z, w->skip);

  // (13)
  crv_tensor_copy(w->skip, z);
  crv_tensor_leaky_relu(z, 0.2f);
  crv_tensor_pad(z, 6, 0);
  crv_tensor_conv1d(z, w->decoder_net_13_aligned_branches_0_net_1_weight, 1, 3);
  crv_tensor_leaky_relu(z, 0.2f);
  crv_tensor_conv1d(z, w->decoder_net_13_aligned_branches_0_net_3_weight, 1, 1);
  crv_tensor_tadd(z, w->skip);

  // (14)
  crv_tensor_copy(w->skip, z);
  crv_tensor_leaky_relu(z, 0.2f);
  crv_tensor_pad(z, 18, 0);
  crv_tensor_conv1d(z, w->decoder_net_14_aligned_branches_0_net_1_weight, 1, 9);
  crv_tensor_leaky_relu(z, 0.2f);
  crv_tensor_conv1d(z, w->decoder_net_14_aligned_branches_0_net_3_weight, 1, 1);
  crv_tensor_tadd(z, w->skip);

  // (15)
  crv_tensor_snake(z, w->net_15_alpha);

  // (16)
  crv_tensor_pad(z, 1, 0);
  crv_tensor_conv_transpose1d(z, w->net_16_weight, 4, 1);
  crv_tensor_trunc(z, 4, 4);

  // (17)
  crv_tensor_copy(w->skip, z);
  crv_tensor_leaky_relu(z, 0.2f);
  crv_tensor_pad(z, 2, 0);
  crv_tensor_conv1d(z, w->decoder_net_17_aligned_branches_0_net_1_weight, 1, 1);
  crv_tensor_leaky_relu(z, 0.2f);
  crv_tensor_conv1d(z, w->decoder_net_17_aligned_branches_0_net_3_weight, 1, 1);
  crv_tensor_tadd(z, w->skip);

  // (18)
  crv_tensor_copy(w->skip, z);
  crv_tensor_leaky_relu(z, 0.2f);
  crv_tensor_pad(z, 6, 0);
  crv_tensor_conv1d(z, w->decoder_net_18_aligned_branches_0_net_1_weight, 1, 3);
  crv_tensor_leaky_relu(z, 0.2f);
  crv_tensor_conv1d(z, w->decoder_net_18_aligned_branches_0_net_3_weight, 1, 1);
  crv_tensor_tadd(z, w->skip);

  // (19)
  crv_tensor_copy(w->skip, z);
  crv_tensor_leaky_relu(z, 0.2f);
  crv_tensor_pad(z, 18, 0);
  crv_tensor_conv1d(z, w->decoder_net_19_aligned_branches_0_net_1_weight, 1, 9);
  crv_tensor_leaky_relu(z, 0.2f);
  crv_tensor_conv1d(z, w->decoder_net_19_aligned_branches_0_net_3_weight, 1, 1);
  crv_tensor_tadd(z, w->skip);

  // (20)
  crv_tensor_snake(z, w->net_20_alpha);

  // (21)
  crv_tensor_pad(z, 6, 0);
  crv_tensor_conv1d(z, w->net_21_weight, 1, 1);

  tensor_t* amplitude = w->skip;

  crv_tensor_split(amplitude, z);
  crv_tensor_sigmoid(amplitude);
  crv_tensor_tmul(z, amplitude);

  crv_tensor_tanh(z);

  crv_tensor_reshape(z, CRV_TPL(1, 16, 128)); 
  crv_tensor_tmul(z, w->mask);
  crv_tensor_pad(z, 32, 0);
  crv_tensor_conv1d(z, w->pqmf_inverse_conv_weight, 1, 1);
  crv_tensor_mul(z, 16);
  crv_tensor_flip(z, 1);
  crv_tensor_permute(z, CRV_TPL(0, 2, 1));
  crv_tensor_reshape(z, CRV_TPL(1, 128, 1, 16));
  crv_tensor_permute(z, CRV_TPL(0, 2, 1, 3));
  crv_tensor_reshape(z, CRV_TPL(1, 1, 2048));
}

void v2_cache_slice(tensor_t* cache, tensor_t* input) {
  size_t input_size = crv_tensor_get_last_dim_size(input);
  size_t cache_size = crv_tensor_get_last_dim_size(cache);

  size_t start = input_size - cache_size;
  size_t cpy_count = input_size - start;

  CRV_DO_INTERNAL(
    crv_tensor_validate(input);
    crv_tensor_validate(cache);

    assert(input->rank == cache->rank);
    assert(input->dims[0] == cache->dims[0]);
    assert(input->dims[1] == cache->dims[1]);
    assert(input_size >= cache_size);
  );

  size_t strides[CRV_MAX_RANK];
  crv_tensor_get_strides(input, &strides[0]);

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
  size_t cache_size = crv_tensor_get_last_dim_size(cache);
  crv_tensor_cat(input, cache, crv_tensor_get_last_dim_index(input), CRV_FRONT);
  v2_cache_slice(cache, input);
  crv_tensor_trunc(input, 0, cache_size);
}

void v2_cached_conv1d(tensor_t* input, tensor_t* weights, tensor_t* cache, size_t stride, size_t dilation) {
  crv_tensor_cat(input, cache, crv_tensor_get_last_dim_index(input), CRV_FRONT);
  v2_cache_slice(cache, input);
  crv_tensor_conv1d(input, weights, stride, dilation);
}

void v2_cached_conv_transpose1d(tensor_t* input, tensor_t* weights, tensor_t* cache, size_t stride, size_t dilation) {
  crv_tensor_conv_transpose1d(input, weights, stride, dilation);

  size_t cache_size = crv_tensor_get_last_dim_size(cache);

  CRV_DO_INTERNAL(
    crv_tensor_validate(input);
    crv_tensor_validate(cache);

    assert(input->rank == cache->rank);
    assert(input->dims[0] == cache->dims[0]);
    assert(input->dims[1] == cache->dims[1]);

    size_t input_size = crv_tensor_get_last_dim_size(input);
    assert(input_size >= cache_size);
  );

  size_t strides[CRV_MAX_RANK];
  crv_tensor_get_strides(input, &strides[0]);

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
  crv_tensor_trunc(input, 0, cache_size);
}

int v2_load(char** dest, v2_model_t* w, tensor_list_t* list) {
#ifdef MODEL_TEST
  w->noise                                              = crv_tensor_find_in_list(list, "pre_process_latent_noise");
  w->ir_noise                                           = crv_tensor_find_in_list(list, "ir_noise");
#else
  w->noise                                              = crv_tensor_create(dest, CRV_TPL(1, 96, 1), CRV_TENSOR_AUTO_CAP, CRV_NO_SWAP);
  w->ir_noise                                           = crv_tensor_create(dest, CRV_TPL(1, 16, 16, 8), CRV_TENSOR_AUTO_CAP, CRV_NO_SWAP);
#endif
  w->latent_pca                                         = crv_tensor_find_in_list(list, "latent_pca");
  w->latent_mean                                        = crv_tensor_find_in_list(list, "latent_mean");
  w->decoder_net_0_weight                               = crv_tensor_find_in_list(list, "decoder.net.0.weight");
  w->decoder_net_0_cache_pad                            = crv_tensor_find_in_list(list, "decoder.net.0.cache.pad");
  w->decoder_net_1_alpha                                = crv_tensor_find_in_list(list, "decoder.net.1.alpha");
  w->decoder_net_2_cache                                = crv_tensor_find_in_list(list, "decoder.net.2.cache");
  w->decoder_net_2_weight                               = crv_tensor_find_in_list(list, "decoder.net.2.weight");
  w->decoder_net_3_aligned_paddings_1_pad               = crv_tensor_find_in_list(list, "decoder.net.3.aligned.paddings.1.pad");
  w->decoder_net_3_aligned_branches_0_net_1_cache_pad   = crv_tensor_find_in_list(list, "decoder.net.3.aligned.branches.0.net.1.cache.pad");
  w->decoder_net_3_aligned_branches_0_net_1_weight      = crv_tensor_find_in_list(list, "decoder.net.3.aligned.branches.0.net.1.weight");
  w->decoder_net_3_aligned_branches_0_net_3_weight      = crv_tensor_find_in_list(list, "decoder.net.3.aligned.branches.0.net.3.weight");
  w->decoder_net_4_aligned_paddings_1_pad               = crv_tensor_find_in_list(list, "decoder.net.4.aligned.paddings.1.pad");
  w->decoder_net_4_aligned_branches_0_net_1_cache_pad   = crv_tensor_find_in_list(list, "decoder.net.4.aligned.branches.0.net.1.cache.pad");
  w->decoder_net_4_aligned_branches_0_net_1_weight      = crv_tensor_find_in_list(list, "decoder.net.4.aligned.branches.0.net.1.weight");
  w->decoder_net_4_aligned_branches_0_net_3_weight      = crv_tensor_find_in_list(list, "decoder.net.4.aligned.branches.0.net.3.weight");
  w->decoder_net_5_alpha                                = crv_tensor_find_in_list(list, "decoder.net.5.alpha");
  w->decoder_net_6_cache                                = crv_tensor_find_in_list(list, "decoder.net.6.cache");
  w->decoder_net_6_weight                               = crv_tensor_find_in_list(list, "decoder.net.6.weight");
  w->decoder_net_7_aligned_paddings_1_pad               = crv_tensor_find_in_list(list, "decoder.net.7.aligned.paddings.1.pad");
  w->decoder_net_7_aligned_branches_0_net_1_cache_pad   = crv_tensor_find_in_list(list, "decoder.net.7.aligned.branches.0.net.1.cache.pad");
  w->decoder_net_7_aligned_branches_0_net_1_weight      = crv_tensor_find_in_list(list, "decoder.net.7.aligned.branches.0.net.1.weight");
  w->decoder_net_7_aligned_branches_0_net_3_weight      = crv_tensor_find_in_list(list, "decoder.net.7.aligned.branches.0.net.3.weight");
  w->decoder_net_8_aligned_paddings_1_pad               = crv_tensor_find_in_list(list, "decoder.net.8.aligned.paddings.1.pad");
  w->decoder_net_8_aligned_branches_0_net_1_cache_pad   = crv_tensor_find_in_list(list, "decoder.net.8.aligned.branches.0.net.1.cache.pad");
  w->decoder_net_8_aligned_branches_0_net_1_weight      = crv_tensor_find_in_list(list, "decoder.net.8.aligned.branches.0.net.1.weight");
  w->decoder_net_8_aligned_branches_0_net_3_weight      = crv_tensor_find_in_list(list, "decoder.net.8.aligned.branches.0.net.3.weight");
  w->decoder_net_9_aligned_paddings_1_pad               = crv_tensor_find_in_list(list, "decoder.net.9.aligned.paddings.1.pad");
  w->decoder_net_9_aligned_branches_0_net_1_cache_pad   = crv_tensor_find_in_list(list, "decoder.net.9.aligned.branches.0.net.1.cache.pad");
  w->decoder_net_9_aligned_branches_0_net_1_weight      = crv_tensor_find_in_list(list, "decoder.net.9.aligned.branches.0.net.1.weight");
  w->decoder_net_9_aligned_branches_0_net_3_weight      = crv_tensor_find_in_list(list, "decoder.net.9.aligned.branches.0.net.3.weight");
  w->decoder_net_10_alpha                               = crv_tensor_find_in_list(list, "decoder.net.10.alpha");
  w->decoder_net_11_cache                               = crv_tensor_find_in_list(list, "decoder.net.11.cache");
  w->decoder_net_11_weight                              = crv_tensor_find_in_list(list, "decoder.net.11.weight");
  w->decoder_net_12_aligned_paddings_1_pad              = crv_tensor_find_in_list(list, "decoder.net.12.aligned.paddings.1.pad");
  w->decoder_net_12_aligned_branches_0_net_1_cache_pad  = crv_tensor_find_in_list(list, "decoder.net.12.aligned.branches.0.net.1.cache.pad");
  w->decoder_net_12_aligned_branches_0_net_1_weight     = crv_tensor_find_in_list(list, "decoder.net.12.aligned.branches.0.net.1.weight");
  w->decoder_net_12_aligned_branches_0_net_3_weight     = crv_tensor_find_in_list(list, "decoder.net.12.aligned.branches.0.net.3.weight");
  w->decoder_net_13_aligned_paddings_1_pad              = crv_tensor_find_in_list(list, "decoder.net.13.aligned.paddings.1.pad");
  w->decoder_net_13_aligned_branches_0_net_1_cache_pad  = crv_tensor_find_in_list(list, "decoder.net.13.aligned.branches.0.net.1.cache.pad");
  w->decoder_net_13_aligned_branches_0_net_1_weight     = crv_tensor_find_in_list(list, "decoder.net.13.aligned.branches.0.net.1.weight");
  w->decoder_net_13_aligned_branches_0_net_3_weight     = crv_tensor_find_in_list(list, "decoder.net.13.aligned.branches.0.net.3.weight");
  w->decoder_net_14_aligned_paddings_1_pad              = crv_tensor_find_in_list(list, "decoder.net.14.aligned.paddings.1.pad");
  w->decoder_net_14_aligned_branches_0_net_1_cache_pad  = crv_tensor_find_in_list(list, "decoder.net.14.aligned.branches.0.net.1.cache.pad");
  w->decoder_net_14_aligned_branches_0_net_1_weight     = crv_tensor_find_in_list(list, "decoder.net.14.aligned.branches.0.net.1.weight");
  w->decoder_net_14_aligned_branches_0_net_3_weight     = crv_tensor_find_in_list(list, "decoder.net.14.aligned.branches.0.net.3.weight");
  w->decoder_net_15_alpha                               = crv_tensor_find_in_list(list, "decoder.net.15.alpha");
  w->decoder_net_16_cache                               = crv_tensor_find_in_list(list, "decoder.net.16.cache");
  w->decoder_net_16_weight                              = crv_tensor_find_in_list(list, "decoder.net.16.weight");
  w->decoder_net_17_aligned_paddings_1_pad              = crv_tensor_find_in_list(list, "decoder.net.17.aligned.paddings.1.pad");
  w->decoder_net_17_aligned_branches_0_net_1_cache_pad  = crv_tensor_find_in_list(list, "decoder.net.17.aligned.branches.0.net.1.cache.pad");
  w->decoder_net_17_aligned_branches_0_net_1_weight     = crv_tensor_find_in_list(list, "decoder.net.17.aligned.branches.0.net.1.weight");
  w->decoder_net_17_aligned_branches_0_net_3_weight     = crv_tensor_find_in_list(list, "decoder.net.17.aligned.branches.0.net.3.weight");
  w->decoder_net_18_aligned_paddings_1_pad              = crv_tensor_find_in_list(list, "decoder.net.18.aligned.paddings.1.pad");
  w->decoder_net_18_aligned_branches_0_net_1_cache_pad  = crv_tensor_find_in_list(list, "decoder.net.18.aligned.branches.0.net.1.cache.pad");
  w->decoder_net_18_aligned_branches_0_net_1_weight     = crv_tensor_find_in_list(list, "decoder.net.18.aligned.branches.0.net.1.weight");
  w->decoder_net_18_aligned_branches_0_net_3_weight     = crv_tensor_find_in_list(list, "decoder.net.18.aligned.branches.0.net.3.weight");
  w->decoder_net_19_aligned_paddings_1_pad              = crv_tensor_find_in_list(list, "decoder.net.19.aligned.paddings.1.pad");
  w->decoder_net_19_aligned_branches_0_net_1_cache_pad  = crv_tensor_find_in_list(list, "decoder.net.19.aligned.branches.0.net.1.cache.pad");
  w->decoder_net_19_aligned_branches_0_net_1_weight     = crv_tensor_find_in_list(list, "decoder.net.19.aligned.branches.0.net.1.weight");
  w->decoder_net_19_aligned_branches_0_net_3_weight     = crv_tensor_find_in_list(list, "decoder.net.19.aligned.branches.0.net.3.weight");
  w->decoder_net_20_alpha                               = crv_tensor_find_in_list(list, "decoder.net.20.alpha");
  w->decoder_noise_module_net_0_cache_pad               = crv_tensor_find_in_list(list, "decoder.noise_module.net.0.cache.pad");
  w->decoder_noise_module_net_0_weight                  = crv_tensor_find_in_list(list, "decoder.noise_module.net.0.weight");
  w->decoder_noise_module_net_1_alpha                   = crv_tensor_find_in_list(list, "decoder.noise_module.net.1.alpha");
  w->decoder_noise_module_net_2_cache_pad               = crv_tensor_find_in_list(list, "decoder.noise_module.net.2.cache.pad");
  w->decoder_noise_module_net_2_weight                  = crv_tensor_find_in_list(list, "decoder.noise_module.net.2.weight");
  w->decoder_noise_module_net_3_alpha                   = crv_tensor_find_in_list(list, "decoder.noise_module.net.3.alpha");
  w->decoder_noise_module_net_4_cache_pad               = crv_tensor_find_in_list(list, "decoder.noise_module.net.4.cache.pad");
  w->decoder_noise_module_net_4_weight                  = crv_tensor_find_in_list(list, "decoder.noise_module.net.4.weight");
  w->decoder_waveform_module_cache_pad                  = crv_tensor_find_in_list(list, "decoder.waveform_module.cache.pad");
  w->decoder_waveform_module_weight                     = crv_tensor_find_in_list(list, "decoder.waveform_module.weight");
  w->pqmf_inverse_conv_cache_pad                        = crv_tensor_find_in_list(list, "pqmf.inverse_conv.cache.pad");
  w->pqmf_inverse_conv_weight                           = crv_tensor_find_in_list(list, "pqmf.inverse_conv.weight");

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

  w->skip = crv_tensor_create(dest, CRV_TPL(8 * 2048), CRV_TENSOR_AUTO_CAP, CRV_SWAP);
  w->zeros = crv_tensor_create(dest, CRV_TPL(1, 16, 16, 5, 1), CRV_TENSOR_AUTO_CAP, CRV_SWAP);
  w->mask = crv_tensor_create(dest, CRV_TPL(1, 16, 128), CRV_TENSOR_AUTO_CAP, CRV_SWAP);
  w->hann = crv_tensor_create(dest, CRV_TPL(8), CRV_TENSOR_AUTO_CAP, CRV_SWAP);
  w->scratch_1 = crv_tensor_create(dest, CRV_TPL(2 * 8192), CRV_TENSOR_AUTO_CAP, CRV_SWAP);
  w->scratch_2 = crv_tensor_create(dest, CRV_TPL(16 * 8192), CRV_TENSOR_AUTO_CAP, CRV_SWAP);

  assert(w->skip != NULL);
  assert(w->zeros != NULL);
  assert(w->mask!= NULL);
  assert(w->hann != NULL);
  assert(w->scratch_1 != NULL);
  assert(w->scratch_2 != NULL);

  crv_tensor_hann(w->hann);
  crv_tensor_fill(w->mask, 1.f);

  size_t channels = w->mask->dims[1];
  size_t len = w->mask->dims[2];
  
  for (size_t i = 1; i < channels; i += 2) {
    for (size_t j = 0; j < len; j += 2) {
      size_t idx = i * len + j; 
      w->mask->data[idx] = -1; 
    }
  }

  w->latent_pca->swap = (float*)*dest;
  *dest += w->latent_pca->count * sizeof(float);

  crv_tensor_unsqueeze(w->latent_pca, w->latent_pca->rank);
  crv_tensor_transpose(w->latent_pca, 0, 1);
  crv_tensor_unsqueeze(w->latent_mean, 0);
  crv_tensor_unsqueeze(w->latent_mean, w->latent_mean->rank);

  return MODEL_LOAD_SUCCESS;
}

void v2_noise_generator(tensor_t* input, v2_model_t* w) {
  // TODO(luca): verify that we actually need this amp stuff as it seems to not
  // really be contributing anything.

  v2_cached_conv1d(input, w->decoder_noise_module_net_0_weight, w->decoder_noise_module_net_0_cache_pad, 2, 1);

  crv_tensor_snake(input, w->decoder_noise_module_net_1_alpha);
  v2_cached_conv1d(input, w->decoder_noise_module_net_2_weight, w->decoder_noise_module_net_2_cache_pad, 2, 1);
  crv_tensor_snake(input, w->decoder_noise_module_net_3_alpha);
  v2_cached_conv1d(input, w->decoder_noise_module_net_4_weight, w->decoder_noise_module_net_4_cache_pad, 2, 1);
  crv_tensor_add(input, -5.f);

  crv_tensor_sigmoid(input);
  crv_tensor_pow(input, 2.3f);
  crv_tensor_mul(input, 2.f);
  crv_tensor_add(input, 1e-7f);
  crv_tensor_permute(input, CRV_TPL(0, 2, 1));
  crv_tensor_reshape(input, CRV_TPL(1, 16, 16, 5));
  crv_tensor_unsqueeze(input, input->rank);
  crv_tensor_cat(input, w->zeros, input->rank - 1, CRV_BACK);
  crv_tensor_irfft(input);
  crv_tensor_roll(input, 4, input->rank - 1);
  crv_tensor_tmul_last_dim(input, w->hann);
  crv_tensor_roll(input, -4, input->rank - 1);

  tensor_t* ir_noise = w->scratch_2;
  crv_tensor_copy(ir_noise, w->ir_noise);
  crv_tensor_pad(ir_noise, 0, 8);
  crv_tensor_pad(input, 8, 0);
  crv_tensor_rfft(ir_noise);
  crv_tensor_rfft(input);
  crv_tensor_tmul_last_dim(input, ir_noise);
  crv_tensor_irfft(input);
   
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

  crv_tensor_permute(input, CRV_TPL(0, 2, 1, 3));
  crv_tensor_reshape(input, CRV_TPL(1, 16, 128));
}

void v2_block(tensor_t* input, tensor_t* skip, tensor_t* w0, tensor_t* w1, tensor_t* c0, tensor_t* c1, size_t dilation) {
  crv_tensor_copy(skip, input);
  v2_cached_pad(skip, c0);
  crv_tensor_leaky_relu(input, 0.2);
  v2_cached_conv1d(input, w0, c1, 1, dilation);
  crv_tensor_leaky_relu(input, 0.2);
  crv_tensor_conv1d(input, w1, 1, 1);
  crv_tensor_tadd(input, skip);
}

void v2_decode(tensor_t* z, v2_model_t* w) {
#ifndef MODEL_TEST
  crv_tensor_randn(w->noise);
  crv_tensor_randn(w->ir_noise);
#endif

  crv_tensor_cat(z, w->noise, 1, CRV_BACK);
  crv_tensor_conv1d(z, w->latent_pca, 1, 1);
  crv_tensor_tadd(z, w->latent_mean);

  v2_cached_conv1d(z, w->decoder_net_0_weight, w->decoder_net_0_cache_pad, 1, 1);
  crv_tensor_snake(z, w->decoder_net_1_alpha);
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

  crv_tensor_snake(z, w->decoder_net_5_alpha);
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

  crv_tensor_snake(z, w->decoder_net_10_alpha);
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

  crv_tensor_snake(z, w->decoder_net_15_alpha);
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

  crv_tensor_snake(z, w->decoder_net_20_alpha);

  tensor_t* noise = w->scratch_1;
  crv_tensor_copy(noise, z);

  v2_noise_generator(noise, w);

  // Waveform module
  v2_cached_conv1d(z, w->decoder_waveform_module_weight, w->decoder_waveform_module_cache_pad, 1, 1);

  // Amplitude modulation
  tensor_t* amp = w->scratch_2;
  crv_tensor_split(amp, z);
  crv_tensor_sigmoid(amp);
  crv_tensor_tmul(z, amp);
  crv_tensor_tadd(z, noise); 
  crv_tensor_tanh(z);

  // Post
  crv_tensor_reshape(z, CRV_TPL(1, 16, 128)); 
  crv_tensor_tmul(z, w->mask);
  v2_cached_conv1d(z, w->pqmf_inverse_conv_weight, w->pqmf_inverse_conv_cache_pad, 1, 1);
  crv_tensor_mul(z, 16);
  crv_tensor_flip(z, 1);
  crv_tensor_permute(z, CRV_TPL(0, 2, 1));
  crv_tensor_reshape(z, CRV_TPL(1, 128, 1, 16));
  crv_tensor_permute(z, CRV_TPL(0, 2, 1, 3));
  crv_tensor_reshape(z, CRV_TPL(1, 1, 2048));
}

int model_load(char** dest, char* src, model_t* model) {
  int result = MODEL_LOAD_ERROR;

  model_get_header_from_memory(&model->header, src); 
  header_t* header = &model->header;
  tensor_list_t* list = crv_tensor_load_from_memory(dest, src + sizeof(header_t), header->num_tensors);

  if (list) {
    switch (header->config) {
      case 1:
        result = v1_load(dest, &model->v1, list);
        break;

      case 2:
        result = v2_load(dest, &model->v2, list);
        break;
    }

  } else {
    fprintf(stderr, "Error loading tensor list.\n");
    result = MODEL_LOAD_ERROR;
  }

  return result;
}

// TODO(luca): model should be first and tensor second.
void model_decode(tensor_t* z, model_t* model) {
  header_t* header = &model->header;

  switch (header->config) {
    case 1:
      v1_decode(z, &model->v1);
      break;

    case 2:
      v2_decode(z, &model->v2);
      break;
  }
}

#endif // MODEL_H
