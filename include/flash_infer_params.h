#pragma once

#include <cstdint>
#include "cuda_runtime.h"

struct FlashInferParams {
  int32_t batch_size;
  int32_t seqlen_q;
  int32_t seqlen_kv;
  int32_t head_num;
  int32_t head_dims_q;
  int32_t head_dims_kv;
  void *Q_ptr = nullptr;
  void *K_ptr = nullptr;
  void *V_ptr = nullptr;
  void *output_ptr = nullptr;
  void *output_host = nullptr;
  float softmax_scale;
  float softmax_scale_log2;
  bool is_bf16;
  cudaStream_t stream = nullptr;
  // int32_t n_block_kv;

  FlashInferParams(const int32_t batch_size, const int32_t seqlen_q, const int32_t seqlen_kv, const int32_t head_num,
                   const int32_t head_dims_q, const int32_t head_dims_kv, float softmax_scale, float softmax_scale_log2,
                   bool is_bf16 = false, void *Q_ptr = nullptr, void *K_ptr = nullptr, void *V_ptr = nullptr, 
                   void *output_ptr = nullptr, void *output_host = nullptr, cudaStream_t stream = nullptr) {
    this->batch_size = batch_size;
    this->seqlen_q = seqlen_q;
    this->seqlen_kv = seqlen_kv;
    this->head_num = head_num;
    this->head_dims_q = head_dims_q;
    this->head_dims_kv = head_dims_kv;
    this->Q_ptr = Q_ptr;
    this->K_ptr = K_ptr;
    this->V_ptr = V_ptr;
    this->output_ptr = output_ptr;
    this->output_host = output_host;
    this->softmax_scale = softmax_scale;
    this->softmax_scale_log2 = softmax_scale_log2;
    this->is_bf16 = is_bf16;
    this->stream = stream;
  }

  FlashInferParams() = default;
};