#include "utils/cuda_utils.h"
using namespace cute;

int main() {

  using Element = cutlass::half_t;
  const int batch_size = 1;
  const int seqlen_q = 400;
  const int seqlen_kv = 32768;
  const int n_heads = 8;
  const int head_dims_q = 32;
  const int head_dims_kv = 32;
  const int n_test = 20;

  // init 
  const int n_bytes_output = batch_size * seqlen_q * n_heads * head_dims_q * sizeof(Element);
  FlashInferParams params(batch_size, seqlen_q, seqlen_kv, n_heads, head_dims_q, head_dims_kv, 0.0, 0.0);
  prepare_input<Element>(params);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // warm up
  run_flash_infer<FlashInferTraits<32, 128, 128, 4, Element>>(params);
  cudaDeviceSynchronize();

  cudaEventRecord(start, params.stream);
  for (int i = 0; i < n_test; i++) {
    run_flash_infer<FlashInferTraits<32, 128, 128, 4, Element>>(params);
  }
  cudaEventRecord(stop, params.stream);
  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  // printf("Time: %f ms\n", ms / n_test);
  cudaMemcpy(params.output_host, params.output_ptr, n_bytes_output, cudaMemcpyDeviceToHost);

  std::string file_ref = "ref_val.bin";
  // read from bin file
  FILE *fp = fopen(file_ref.c_str(), "rb");
  if (fp == NULL) {
    printf("Error: open file %s failed\n", file_ref.c_str());
    return -1;
  }

  int ref_size = seqlen_q * n_heads * head_dims_q;
  std::vector<float> ref_val(ref_size);
  fread((void *)ref_val.data(), sizeof(float), ref_size, fp);
  fclose(fp);
  bool is_same = true;
  Element* O_host = (Element*)params.output_host;
  for (int i = 0; i < seqlen_q && is_same; i++) {
    for (int i_head = 0; i_head < n_heads && is_same; i_head++) {
      for (int j = 0; j < head_dims_q && is_same; j++) {
        int index = i * n_heads * head_dims_q + i_head * head_dims_q + j;
        if (fabs(float(O_host[index]) - ref_val[index]) > 0.01) {
          printf("Error: i: %d, j: %d, C: %f, C_ref: %f\n", i, j, float(O_host[index]), ref_val[index]);
          is_same = false;
        }
      }
    }
  }
  
  printf("Q: %d K: %d V: %d Head: %d n_emb %d, latency: %f ms\n", seqlen_q, seqlen_kv, seqlen_kv, n_heads, head_dims_q, ms / n_test);
  printf("\noutput data sample:\n");
  printf("O_host[0] C: %f, C_ref: %f\n", float(O_host[0]), float(ref_val[0]));
  printf("O_host[13] C: %f, C_ref: %f\n", float(O_host[13]), float(ref_val[13]));

  if (is_same) {
    printf("Result is correct!\n");
  } else {
    printf("Result is wrong!\n");
  }

  // free
  if (params.Q_ptr) {
    cudaFree(params.Q_ptr);
    params.Q_ptr = nullptr;
  }
  if (params.K_ptr) {
    cudaFree(params.K_ptr);
    params.K_ptr = nullptr;
  }
  if (params.V_ptr) {
    cudaFree(params.V_ptr);
    params.V_ptr = nullptr;
  }
  if (params.output_ptr) {
    cudaFree(params.output_ptr);
    params.output_ptr = nullptr;
  }
  if (params.stream) {
    cudaStreamDestroy(params.stream);
    params.stream = nullptr;
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}