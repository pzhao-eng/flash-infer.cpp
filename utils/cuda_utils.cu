#include "cuda_utils.h"

void output_as_bin(const char *filename, const char *data, const int n) {
  FILE *fp = fopen(filename, "wb");
  fwrite(data, sizeof(char), n, fp);
  fclose(fp);
}

template <typename kernel_traits>
void run_flash_infer(const FlashInferParams params) {
  constexpr int  kNThreads = kernel_traits::kNThreads;
  constexpr int kBlockM = kernel_traits::kBlockM;
  constexpr int kBlockN = kernel_traits::kBlockN;
  constexpr int kSmemSize = kernel_traits::kSmemSize;
  const int seqlen_q = params.seqlen_q;
  const int seqlen_kv = params.seqlen_kv;
  const int n_heads = params.head_num;

  dim3 block(kNThreads);
  int grid_x = (seqlen_q + kBlockM - 1) / kBlockM;
  dim3 grid(grid_x, n_heads);
  const int n_block_kv = cute::ceil_div(seqlen_kv, kBlockN);
  bool is_even_MN = seqlen_q % kBlockM == 0 && seqlen_kv % kBlockN == 0;
  if (is_even_MN) {
    flash_infer_kernel<kernel_traits, true><<<grid, block, kSmemSize>>>(params, n_block_kv);
  } else {
    flash_infer_kernel<kernel_traits, false><<<grid, block, kSmemSize>>>(params, n_block_kv);
  }
}

template <typename kernel_traits>
void run_flash_split_kv_infer(const FlashInferParams params) {
  constexpr int  kNThreads = kernel_traits::kNThreads;
  constexpr int kBlockM = kernel_traits::kBlockM;
  constexpr int kBlockN = kernel_traits::kBlockN;
  constexpr int kSmemSize = kernel_traits::kSmemSize;
  constexpr int kSplitKV = kernel_traits::kSplitKV;
  const int seqlen_q = params.seqlen_q;
  const int seqlen_kv = params.seqlen_kv / kSplitKV;
  const int n_heads = params.head_num;
  if (seqlen_kv % kBlockN != 0) {
    printf("Error: seqlen_kv %d is not divisible by %d\n", seqlen_kv, kBlockN);
    return;
  }

  dim3 block(kNThreads);
  int grid_x = (seqlen_q + kBlockM - 1) / kBlockM;
  dim3 grid(grid_x, n_heads, kSplitKV);
  const int n_block_kv = cute::ceil_div(seqlen_kv, kBlockN);
  bool is_even_MN = seqlen_q % kBlockM == 0 && seqlen_kv % kBlockN == 0;
  if (is_even_MN) {
    flash_infer_split_kv_kernel<kernel_traits, true><<<grid, block, kSmemSize, params.stream>>>(params, n_block_kv);
  } else {
    flash_infer_split_kv_kernel<kernel_traits, false><<<grid, block, kSmemSize, params.stream>>>(params, n_block_kv);
  }
  int dev_id = 0;
  cudaGetDevice(&dev_id);
  int num_sm = 0;
  cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id);
  printf("num_sm: %d, dev_id: %d\n", num_sm, dev_id);

  // constexpr int kBlockMCombine = 32;
  // dim3 grid_combine((params.head_num * params.seqlen_q + kBlockMCombine - 1) / kBlockMCombine);
  // if (kSplitKV <= 2) {
  //   combine_attn_seqk_parallel<kernel_traits, kBlockMCombine, 1, true><<<grid_combine, kNThreads, 0, params.stream>>>(params);
  // } else if (kSplitKV <= 4) {
  //   combine_attn_seqk_parallel<kernel_traits, kBlockMCombine, 2, true><<<grid_combine, kNThreads, 0, params.stream>>>(params);
  // } else if (kSplitKV <= 8) {
  //   combine_attn_seqk_parallel<kernel_traits, kBlockMCombine, 3, true><<<grid_combine, kNThreads, 0, params.stream>>>(params);
  // } else {
  //   printf("Error: num_splits %d is not supported\n", kSplitKV);
  // }
}

template <typename Element, int SplitKV = 1>
void prepare_input(FlashInferParams &params) {

  const int seqlen_q = params.seqlen_q;
  const int seqlen_kv = params.seqlen_kv;
  const int n_heads = params.head_num;
  const int head_dims_q = params.head_dims_q;
  const int head_dims_kv = params.head_dims_kv;
  const int dims_q = seqlen_q * n_heads * head_dims_q;
  const int dims_kv = seqlen_kv * n_heads * head_dims_kv;
  const int n_bytes_q = dims_q * sizeof(Element);
  const int n_bytes_kv = dims_kv * sizeof(Element);
  const int n_bytes_output = dims_q * sizeof(Element);
  const int n_bytes_lse_accum = SplitKV * seqlen_q * n_heads * sizeof(float);
  Element *Q_host = (Element*)malloc(n_bytes_q);
  Element *K_host = (Element*)malloc(n_bytes_kv);
  Element *V_host = (Element*)malloc(n_bytes_kv);
  Element *O_host = (Element*)malloc(n_bytes_output);
  Element *O_host_ref = (Element*)malloc(n_bytes_output);

  void *Q_dev = nullptr;
  void *K_dev = nullptr;
  void *V_dev = nullptr;
  void *O_dev = nullptr;
  void *O_acc = nullptr;
  void *lse_accum = nullptr;
  cudaMalloc(&Q_dev, n_bytes_q);
  cudaMalloc(&K_dev, n_bytes_kv);
  cudaMalloc(&V_dev, n_bytes_kv);
  cudaMalloc(&O_acc, n_bytes_output * SplitKV);
  cudaMalloc(&O_dev, n_bytes_output);
  cudaMalloc(&lse_accum, n_bytes_lse_accum);

  float softmax_scale = 1.0 / sqrtf(float(head_dims_q));
  float softmax_scale_log2 = softmax_scale * M_LOG2E;
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);
  
  params.Q_ptr = Q_dev;
  params.K_ptr = K_dev;
  params.V_ptr = V_dev;
  params.output_ptr = O_dev;
  params.output_accum_ptr = O_acc;
  params.lse_accum_ptr = lse_accum;
  params.output_host = O_host;
  params.stream = stream;
  params.softmax_scale = softmax_scale;
  params.softmax_scale_log2 = softmax_scale_log2;

  // set rand seed
  srand(0);
  for (int i = 0; i < dims_q; i++) {
    // Q_host[i] = static_cast<Element>(i + 1) * 0.1;
    // gen random float number
    float random = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    Q_host[i] = static_cast<Element>(random);
    int index = rand() % dims_q;
    Q_host[index] = 0.0;
  }
  for (int i = 0; i < dims_kv; i++) {
    // K_host[i] = static_cast<Element>(i + 1) * 0.1;
    // V_host[i] = static_cast<Element>(i + 1) * 0.1;
    float random = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    K_host[i] = static_cast<Element>(random);
    V_host[i] = static_cast<Element>(random);
  }

  for (int i = 0; i < 100; i++) {
    int index = rand() % dims_q;
    Q_host[index] = 0.0;
    int index_b = rand() % dims_kv;
    K_host[index_b] = 0.0;
    V_host[index_b] = 0.0;
  }

  cudaMemcpy(params.Q_ptr, Q_host, n_bytes_q, cudaMemcpyHostToDevice);
  cudaMemcpy(params.K_ptr, K_host, n_bytes_kv, cudaMemcpyHostToDevice);
  cudaMemcpy(params.V_ptr, V_host, n_bytes_kv, cudaMemcpyHostToDevice);
  cudaMemset(params.output_ptr, 0, n_bytes_output);
  cudaMemset(params.output_accum_ptr, 0, n_bytes_output * SplitKV);

  // output as bin file
  output_as_bin("q.bin", (const char*)Q_host, n_bytes_q);
  output_as_bin("k.bin", (const char*)K_host, n_bytes_kv);
  output_as_bin("v.bin", (const char*)V_host, n_bytes_kv);

  printf("input data sample:\n");
  printf("Q_host[0] A[0]: %f A[1]: %f\n", float(Q_host[0]), float(Q_host[1]));
  printf("K_host[0] B[0]: %f B[1]: %f\n", float(K_host[0]), float(K_host[1]));
  printf("V_host[0] V[0]: %f V[1]: %f\n", float(V_host[0]), float(V_host[1]));
  free(Q_host);
  free(K_host);
  free(V_host);
}

template void prepare_input<cutlass::half_t, 1>(FlashInferParams &params);
template void prepare_input<cutlass::half_t, 4>(FlashInferParams &params);
template void prepare_input<cutlass::half_t, 8>(FlashInferParams &params);
// template void run_flash_infer<FlashInferTraits<32, 128, 128, 4, cutlass::half_t, 1>>(const FlashInferParams params);
template void run_flash_split_kv_infer<FlashInferTraits<32, 128, 128, 4, cutlass::half_t, 8>>(const FlashInferParams params);
template void run_flash_split_kv_infer<FlashInferTraits<32, 128, 128, 4, cutlass::half_t, 4>>(const FlashInferParams params);
template void run_flash_split_kv_infer<FlashInferTraits<32, 128, 64, 4, cutlass::half_t, 4>>(const FlashInferParams params);
template void run_flash_split_kv_infer<FlashInferTraits<32, 64, 128, 4, cutlass::half_t, 8>>(const FlashInferParams params);