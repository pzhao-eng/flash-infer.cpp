#pragma once
#include "flash_infer_params.h"
#include "softmax.h"
#include "utils.h"

// A[n_row, num_heads, kHeadDim]
// B[n_col, num_heads, kHeadDim]
// V[n_col, num_heads, kHeadDim]
template <typename kernel_Traits, bool is_even_MN>
__global__ void flash_infer_kernel(FlashInferParams params, const int n_block) {
  using Element = typename kernel_Traits::Element;
  using GmemTiledCopyQKV = typename kernel_Traits::GmemTiledCopyQKV;
  using TiledMma = typename kernel_Traits::TiledMma;
  using SmemCopyAtom = typename kernel_Traits::SmemCopyAtom;
  using SmemCopyAtomTransposed = typename kernel_Traits::SmemCopyAtomTransposed;
  using SmemLayoutO = typename kernel_Traits::SmemLayoutO;
  using SmemCopyAtomO = typename kernel_Traits::SmemCopyAtomO;
  using GmemTiledCopyO = typename kernel_Traits::GmemTiledCopyO;
  using SmemLayoutQ = typename kernel_Traits::SmemLayoutQ;
  using SmemLayoutKV = typename kernel_Traits::SmemLayoutKV;
  using SmemLayoutVtransposed = typename kernel_Traits::SmemLayoutVtransposed;
  using SmemLayoutVtransposedNoSwizzle = typename kernel_Traits::SmemLayoutVtransposedNoSwizzle;

  constexpr int kHeadDim = kernel_Traits::kHeadDim;
  constexpr int kBlockM = kernel_Traits::kBlockM;
  constexpr int kBlockN = kernel_Traits::kBlockN;
  constexpr int kNThreads = kernel_Traits::kNThreads;
  const bool Share_Q_K_smem = kernel_Traits::Share_Q_K_smem;

  int b_row = blockIdx.x;
  auto tidx = threadIdx.x;
  int head_idx = blockIdx.y;
  extern __shared__ Element smem[];
  const int n_row= params.seqlen_q;
  const int n_col= params.seqlen_kv;
  const int num_heads = params.head_num;
  const float softmax_scale_log2 = params.softmax_scale_log2;
  const float softmax_scale = params.softmax_scale;


  Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.Q_ptr)),
                          make_shape(n_row, num_heads, Int<kHeadDim>{}), make_stride(num_heads * kHeadDim, Int<kHeadDim>{}, _1{}));
  Tensor gQ = local_tile(mQ(_, head_idx, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(b_row, 0));

  Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.K_ptr)),
                          make_shape(n_col, num_heads, Int<kHeadDim>{}), make_stride(num_heads * kHeadDim, Int<kHeadDim>{}, _1{}));
  Tensor gK = local_tile(mK(_, head_idx, _), Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, 0));

  Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.V_ptr)),
                          make_shape(n_col, num_heads, Int<kHeadDim>{}), make_stride(num_heads * kHeadDim, Int<kHeadDim>{}, _1{}));
  Tensor gV = local_tile(mV(_, head_idx, _), Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, 0));

  Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.output_ptr)),
                          make_shape(n_row, num_heads, Int<kHeadDim>{}), make_stride(num_heads * kHeadDim, Int<kHeadDim>{}, _1{}));
  Tensor gO = local_tile(mO(_, head_idx, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(b_row, 0));

  Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem)), SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(sQ.data() + size(SmemLayoutQ{})), SmemLayoutKV{});
  Tensor sV = make_tensor(sK.data() + size(sK), SmemLayoutKV{});
  Tensor sVt = make_tensor(sV.data(), SmemLayoutVtransposed{});\
  Tensor sVtNoSwizzle = make_tensor(sV.data(), SmemLayoutVtransposedNoSwizzle{});

  // load data from global memory to shared memory
  GmemTiledCopyQKV gmem_tiled_copy_QKV;
  auto gemm_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
  Tensor tQgQ = gemm_thr_copy_QKV.partition_S(gQ); // thread A copy of gQ
  Tensor tQsQ = gemm_thr_copy_QKV.partition_D(sQ);
  Tensor tKgK = gemm_thr_copy_QKV.partition_S(gK);
  Tensor tKsK = gemm_thr_copy_QKV.partition_D(sK);
  Tensor tVgV = gemm_thr_copy_QKV.partition_S(gV);
  Tensor tVsV = gemm_thr_copy_QKV.partition_D(sV);
  Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
  Tensor tQcQ = gemm_thr_copy_QKV.partition_S(cQ);
  Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
  Tensor tKVcKV = gemm_thr_copy_QKV.partition_S(cKV);

  flash::copy_simplify<is_even_MN>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, n_row - b_row * kBlockM);
  flash::copy_simplify<is_even_MN>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, n_col);
  cute::cp_async_fence();
  flash::copy_simplify<is_even_MN>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block - 1), tVsV, tKVcKV, n_col);
  cute::cp_async_fence();

  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tidx);
  Tensor tSrA = thr_mma.partition_fragment_A(sQ); // Source thread register A
  Tensor tSrB = thr_mma.partition_fragment_B(sK);
  Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);
  Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
  clear(acc_s);
  Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
  clear(acc_o);

  auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(tidx);
  Tensor tSsA = smem_thr_copy_A.partition_S(sQ); // Source thread shared A

  auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(tidx);
  Tensor tSsB = smem_thr_copy_B.partition_S(sK);

  auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
  Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

  cute::cp_async_wait<1>();
  __syncthreads();

  flash::gemm_warp(acc_s, tSrA, tSrB, tSsA, tSsB, tiled_mma, smem_tiled_copy_A, smem_tiled_copy_B, smem_thr_copy_A, smem_thr_copy_B);

  flash::Softmax<2 * size<1>(acc_o)> softmax;
  softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/false>(acc_s, acc_o, softmax_scale_log2);
  for (int i_b = n_block - 2; i_b >= 0; --i_b) {
    // prefetch next block of K
    flash::copy_simplify<true>(gmem_tiled_copy_QKV, tKgK(_, _, _, i_b), tKsK, tKVcKV);
    cute::cp_async_fence();

    cute::cp_async_wait<1>();
    __syncthreads();

    // Convert acc_s from fp32 to fp16/bf16
    Tensor rP = flash::convert_type<Element>(acc_s);
    // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
    // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
    Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<TiledMma>(rP.layout()));

    flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

    // prefetch next block of V
    flash::copy_simplify<true>(gmem_tiled_copy_QKV, tVgV(_, _, _, i_b), tVsV, tKVcKV);
    cute::cp_async_fence();

    cute::cp_async_wait<1>();
    __syncthreads();
    clear(acc_s);
    flash::gemm_warp(acc_s, tSrA, tSrB, tSsA, tSsB, tiled_mma, smem_tiled_copy_A, smem_tiled_copy_B, smem_thr_copy_A, smem_thr_copy_B);

    softmax.template softmax_rescale_o</*Is_first=*/false,  /*Check_inf=*/false>(acc_s, acc_o, softmax_scale_log2);
  }
  flash::cp_async_wait<0>();
  __syncthreads();
  // Convert acc_s from fp32 to fp16/bf16
  Tensor rP = flash::convert_type<Element>(acc_s);
  // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
  // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
  Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<TiledMma>(rP.layout()));

  flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

  // // Epilogue
  softmax.template normalize_softmax_lse<false>(acc_o, softmax_scale);
  // Convert acc_o from fp32 to fp16/bf16
  Tensor rO = flash::convert_type<Element>(acc_o);
  Tensor sO = make_tensor(sQ.data(), SmemLayoutO{});    // (SMEM_M,SMEM_N)
  // Partition sO to match the accumulator partitioning
  auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
  auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
  Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
  Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

  // sO has the same size as sQ, so we don't need to sync here.
  if (Share_Q_K_smem) { __syncthreads(); }

  cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

  GmemTiledCopyO gmem_tiled_copy_O;
  auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
  Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
  Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
  Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));
  Tensor tOcO = gmem_thr_copy_O.partition_S(cO);

  __syncthreads();

  Tensor tOrO = make_tensor<Element>(shape(tOgO));
  cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

  flash::copy_simplify<is_even_MN>(gmem_tiled_copy_O, tOrO, tOgO, tOcO, n_row - b_row * kBlockM);
}


// A[n_row, num_heads, kHeadDim]
// B[n_col, num_heads, kHeadDim]
// V[n_col, num_heads, kHeadDim]
template <typename kernel_Traits, bool is_even_MN>
__global__ void flash_infer_split_kv_kernel(FlashInferParams params, const int n_block) {
  using Element = typename kernel_Traits::Element;
  using GmemTiledCopyQKV = typename kernel_Traits::GmemTiledCopyQKV;
  using TiledMma = typename kernel_Traits::TiledMma;
  using SmemCopyAtom = typename kernel_Traits::SmemCopyAtom;
  using SmemCopyAtomTransposed = typename kernel_Traits::SmemCopyAtomTransposed;
  using SmemLayoutO = typename kernel_Traits::SmemLayoutO;
  using SmemCopyAtomO = typename kernel_Traits::SmemCopyAtomO;
  using GmemTiledCopyO = typename kernel_Traits::GmemTiledCopyO;
  using SmemLayoutQ = typename kernel_Traits::SmemLayoutQ;
  using SmemLayoutKV = typename kernel_Traits::SmemLayoutKV;
  using SmemLayoutVtransposed = typename kernel_Traits::SmemLayoutVtransposed;
  using SmemLayoutVtransposedNoSwizzle = typename kernel_Traits::SmemLayoutVtransposedNoSwizzle;
  
  constexpr int kHeadDim = kernel_Traits::kHeadDim;
  constexpr int kBlockM = kernel_Traits::kBlockM;
  constexpr int kBlockN = kernel_Traits::kBlockN;
  constexpr int kNThreads = kernel_Traits::kNThreads;
  constexpr int kSplitKV = kernel_Traits::kSplitKV;
  const bool Share_Q_K_smem = kernel_Traits::Share_Q_K_smem;

  int b_row = blockIdx.x;
  auto tidx = threadIdx.x;
  int head_idx = blockIdx.y;
  int split_idx = blockIdx.z;
  extern __shared__ Element smem[];
  const int n_row= params.seqlen_q;
  const int n_col= params.seqlen_kv / kSplitKV;
  const int num_heads = params.head_num;
  const float softmax_scale_log2 = params.softmax_scale_log2;
  const float softmax_scale = params.softmax_scale;

  Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.Q_ptr)),
                          make_shape(n_row, num_heads, Int<kHeadDim>{}), make_stride(num_heads * kHeadDim, Int<kHeadDim>{}, _1{}));
  Tensor gQ = local_tile(mQ(_, head_idx, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(b_row, 0));

  Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.K_ptr) + split_idx * n_col * num_heads * kHeadDim),
                          make_shape(n_col, num_heads, Int<kHeadDim>{}), make_stride(num_heads * kHeadDim, Int<kHeadDim>{}, _1{}));
  Tensor gK = local_tile(mK(_, head_idx, _), Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, 0));

  Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.V_ptr) + split_idx * n_col * num_heads * kHeadDim),
                          make_shape(n_col, num_heads, Int<kHeadDim>{}), make_stride(num_heads * kHeadDim, Int<kHeadDim>{}, _1{}));
  Tensor gV = local_tile(mV(_, head_idx, _), Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, 0));

  Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.output_accum_ptr) + split_idx * n_row * num_heads * kHeadDim),
                          make_shape(n_row, num_heads, Int<kHeadDim>{}), make_stride(num_heads * kHeadDim, Int<kHeadDim>{}, _1{}));
  Tensor gO = local_tile(mO(_, head_idx, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(b_row, 0));

  Tensor mlse_accum = make_tensor(make_gmem_ptr(reinterpret_cast<float *>(params.lse_accum_ptr)),
                          make_shape(Int<kSplitKV>{}, n_row, num_heads), make_stride(n_row * num_heads, num_heads, _1{}));
  Tensor glse_accum = local_tile(mlse_accum(_, _, head_idx), Shape<Int<kSplitKV>, Int<kBlockM>>{}, make_coord(split_idx, b_row));

  Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem)), SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(sQ.data() + size(SmemLayoutQ{})), SmemLayoutKV{});
  Tensor sV = make_tensor(sK.data() + size(sK), SmemLayoutKV{});
  Tensor sVt = make_tensor(sV.data(), SmemLayoutVtransposed{});\
  Tensor sVtNoSwizzle = make_tensor(sV.data(), SmemLayoutVtransposedNoSwizzle{});

  // load data from global memory to shared memory
  GmemTiledCopyQKV gmem_tiled_copy_QKV;
  auto gemm_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
  Tensor tQgQ = gemm_thr_copy_QKV.partition_S(gQ); // thread A copy of gQ
  Tensor tQsQ = gemm_thr_copy_QKV.partition_D(sQ);
  Tensor tKgK = gemm_thr_copy_QKV.partition_S(gK);
  Tensor tKsK = gemm_thr_copy_QKV.partition_D(sK);
  Tensor tVgV = gemm_thr_copy_QKV.partition_S(gV);
  Tensor tVsV = gemm_thr_copy_QKV.partition_D(sV);
  Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
  Tensor tQcQ = gemm_thr_copy_QKV.partition_S(cQ);
  Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
  Tensor tKVcKV = gemm_thr_copy_QKV.partition_S(cKV);

  flash::copy_simplify<is_even_MN>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, n_row - b_row * kBlockM);
  flash::copy_simplify<is_even_MN>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, n_col);
  cute::cp_async_fence();
  flash::copy_simplify<is_even_MN>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block - 1), tVsV, tKVcKV, n_col);
  cute::cp_async_fence();

  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tidx);
  Tensor tSrA = thr_mma.partition_fragment_A(sQ); // Source thread register A
  Tensor tSrB = thr_mma.partition_fragment_B(sK);
  Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);
  Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
  clear(acc_s);
  Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
  clear(acc_o);

  auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(tidx);
  Tensor tSsA = smem_thr_copy_A.partition_S(sQ); // Source thread shared A

  auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(tidx);
  Tensor tSsB = smem_thr_copy_B.partition_S(sK);

  auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
  Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

  cute::cp_async_wait<1>();
  __syncthreads();

  flash::gemm_warp(acc_s, tSrA, tSrB, tSsA, tSsB, tiled_mma, smem_tiled_copy_A, smem_tiled_copy_B, smem_thr_copy_A, smem_thr_copy_B);

  flash::Softmax<2 * size<1>(acc_o)> softmax;
  softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/false>(acc_s, acc_o, softmax_scale_log2);
  for (int i_b = n_block - 2; i_b >= 0; --i_b) {
    // prefetch next block of K
    flash::copy_simplify<true>(gmem_tiled_copy_QKV, tKgK(_, _, _, i_b), tKsK, tKVcKV);
    cute::cp_async_fence();

    cute::cp_async_wait<1>();
    __syncthreads();

    // Convert acc_s from fp32 to fp16/bf16
    Tensor rP = flash::convert_type<Element>(acc_s);
    // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
    // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
    Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<TiledMma>(rP.layout()));

    flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

    // prefetch next block of V
    flash::copy_simplify<true>(gmem_tiled_copy_QKV, tVgV(_, _, _, i_b), tVsV, tKVcKV);
    cute::cp_async_fence();

    cute::cp_async_wait<1>();
    __syncthreads();
    clear(acc_s);
    flash::gemm_warp(acc_s, tSrA, tSrB, tSsA, tSsB, tiled_mma, smem_tiled_copy_A, smem_tiled_copy_B, smem_thr_copy_A, smem_thr_copy_B);

    softmax.template softmax_rescale_o</*Is_first=*/false,  /*Check_inf=*/false>(acc_s, acc_o, softmax_scale_log2);
  }
  flash::cp_async_wait<0>();
  __syncthreads();
  // Convert acc_s from fp32 to fp16/bf16
  Tensor rP = flash::convert_type<Element>(acc_s);
  // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
  // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
  Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<TiledMma>(rP.layout()));

  flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

  // // Epilogue
  Tensor lse = softmax.template normalize_softmax_lse<false>(acc_o, softmax_scale);
  // Convert acc_o from fp32 to fp16/bf16
  Tensor rO = flash::convert_type<Element>(acc_o);
  Tensor sO = make_tensor(sQ.data(), SmemLayoutO{});    // (SMEM_M,SMEM_N)
  // Partition sO to match the accumulator partitioning
  auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
  auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
  Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
  Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

  // sO has the same size as sQ, so we don't need to sync here.
  if (Share_Q_K_smem) { __syncthreads(); }

  cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

  GmemTiledCopyO gmem_tiled_copy_O;
  auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
  Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
  Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
  Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));
  Tensor tOcO = gmem_thr_copy_O.partition_S(cO);
  Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
  Tensor taccOcO = thr_mma.partition_C(caccO); 
  static_assert(decltype(size<0>(taccOcO))::value == 4);
  // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
  Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
  CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
  if (get<1>(taccOcO_row(0)) == 0) {
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
      const int row = get<0>(taccOcO_row(mi));
      if (row < n_row - b_row * kBlockM) { glse_accum(row) = lse(mi); }
    }
  }

  __syncthreads();

  Tensor tOrO = make_tensor<Element>(shape(tOgO));
  cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

  flash::copy_simplify<is_even_MN>(gmem_tiled_copy_O, tOrO, tOgO, tOcO, n_row - b_row * kBlockM);
}

template<typename Kernel_traits, int kBlockM, int Log_max_splits, bool Is_even_K, typename Params>
inline __global__ void combine_attn_seqk_parallel(const Params &params) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = float;
    constexpr int kMaxSplits = 1 << Log_max_splits;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNThreads = Kernel_traits::kNThreads;
    constexpr int kSplitKV = Kernel_traits::kSplitKV;

    static_assert(kMaxSplits <= 128, "kMaxSplits must be <= 128");
    static_assert(kBlockM == 4 || kBlockM == 8 || kBlockM == 16 || kBlockM == 32, "kBlockM must be 4, 8, 16 or 32");
    static_assert(kNThreads == 128, "We assume that each block has 128 threads");

    // Shared memory.
    // kBlockM + 1 instead of kBlockM to reduce bank conflicts.
    __shared__ ElementAccum sLSE[kMaxSplits][kBlockM + 1];

    // The thread and block index.
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;

    const int row_offset_lse = bidx * kBlockM;
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.lse_accum_ptr) + row_offset_lse),
                                   Shape<Int<kMaxSplits>, Int<kBlockM>>{},
                                   make_stride(params.head_num * params.seqlen_q, _1{}));
    constexpr int kNLsePerThread = (kMaxSplits * kBlockM + kNThreads - 1) / kNThreads;

    // Read the LSE values from gmem and store them in shared memory, then tranpose them.
    constexpr int kRowsPerLoadLSE = kNThreads / kBlockM;
    #pragma unroll
    for (int l = 0; l < kNLsePerThread; ++l) {
        const int row = l * kRowsPerLoadLSE + tidx / kBlockM;
        const int col = tidx % kBlockM;
        ElementAccum lse = (row < kSplitKV && col < params.head_num * params.seqlen_q - bidx * kBlockM) ? gLSEaccum(row, col) : -INFINITY;
        if (row < kMaxSplits) { sLSE[row][col] = lse; }
        // if (bidx == 0 && tidx < 32) { printf("tidx = %d, row = %d, col = %d, lse = %f\n", tidx, row, col, col); }
    }
    // if (bidx == 1 && tidx < 32) { printf("tidx = %d, row_offset_lse = %d, lse = %f\n", tidx, row_offset_lse, lse_accum(0)); }
    __syncthreads();
    Tensor lse_accum = make_tensor<ElementAccum>(Shape<Int<kNLsePerThread>>{});
    constexpr int kRowsPerLoadTranspose = std::min(kRowsPerLoadLSE, kMaxSplits);
    // To make sure that kMaxSplits is within 1 warp: we decide how many elements within kMaxSplits
    // each thread should hold. If kMaxSplits = 16, then each thread holds 2 elements (128 threads,
    // kBlockM rows, so each time we load we can load 128 / kBlockM rows).
    // constexpr int kThreadsPerSplit = kMaxSplits / kRowsPerLoadTranspose;
    // static_assert(kThreadsPerSplit <= 32);
    static_assert(kRowsPerLoadTranspose <= 32);
    static_assert(kNLsePerThread * kRowsPerLoadTranspose <= kMaxSplits);
    #pragma unroll
    for (int l = 0; l < kNLsePerThread; ++l) {
        const int row = l * kRowsPerLoadTranspose + tidx % kRowsPerLoadTranspose;
        const int col = tidx / kRowsPerLoadTranspose;
        lse_accum(l) = (row < kMaxSplits && col < kBlockM) ? sLSE[row][col] : -INFINITY;
        // if (bidx == 0 && tidx < 32) { printf("tidx = %d, row = %d, col = %d, lse = %f\n", tidx, row, col, lse_accum(l)); }
    }

    // // Compute the logsumexp of the LSE along the split dimension.
    // ElementAccum lse_max = lse_accum(0);
    // #pragma unroll
    // for (int l = 1; l < kNLsePerThread; ++l) { lse_max = max(lse_max, lse_accum(l)); }
    // flash::MaxOp<float> max_op;
    // lse_max = flash::Allreduce<kRowsPerLoadTranspose>::run(lse_max, max_op);
    // lse_max = lse_max == -INFINITY ? 0.0f : lse_max;  // In case all local LSEs are -inf
    // float lse_sum = expf(lse_accum(0) - lse_max);
    // #pragma unroll
    // for (int l = 1; l < kNLsePerThread; ++l) { lse_sum += expf(lse_accum(l) - lse_max); }
    // flash::SumOp<float> sum_op;
    // lse_sum = flash::Allreduce<kRowsPerLoadTranspose>::run(lse_sum, sum_op);
    // // For the case where all local lse == -INFINITY, we want to set lse_logsum to INFINITY. Otherwise
    // // lse_logsum is log(0.0) = -INFINITY and we get NaN when we do lse_accum(l) - lse_logsum.
    // ElementAccum lse_logsum = (lse_sum == 0.f || lse_sum != lse_sum) ? INFINITY : logf(lse_sum) + lse_max;
    // // if (bidx == 0 && tidx < 32) { printf("tidx = %d, lse = %f, lse_max = %f, lse_logsum = %f\n", tidx, lse_accum(0), lse_max, lse_logsum); }
    // // Store the scales exp(lse - lse_logsum) in shared memory.
    // #pragma unroll
    // for (int l = 0; l < kNLsePerThread; ++l) {
    //     const int row = l * kRowsPerLoadTranspose + tidx % kRowsPerLoadTranspose;
    //     const int col = tidx / kRowsPerLoadTranspose;
    //     if (row < kSplitKV && col < kBlockM) { sLSE[row][col] = expf(lse_accum(l) - lse_logsum); }
    // }
    // __syncthreads();

    // const int row_offset_oaccum = bidx * kBlockM * params.head_dims_q;
    // Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.output_accum_ptr) + row_offset_oaccum),
    //                              Shape<Int<kBlockM>, Int<kHeadDim>>{},
    //                              Stride<Int<kHeadDim>, _1>{});
    // constexpr int kBlockN = kNThreads / kBlockM;
    // using GmemLayoutAtomOaccum = Layout<Shape<Int<kBlockM>, Int<kBlockN>>, Stride<Int<kBlockN>, _1>>;
    // using GmemTiledCopyOaccum = decltype(
    //     make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
    //                     GmemLayoutAtomOaccum{},
    //                     Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
    // GmemTiledCopyOaccum gmem_tiled_copy_Oaccum;
    // auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    // Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_S(gOaccum);
    // Tensor tOrO = make_tensor<ElementAccum>(shape(tOgOaccum));
    // Tensor tOrOaccum = make_tensor<ElementAccum>(shape(tOgOaccum));
    // clear(tOrO);

    // // Predicates
    // Tensor cOaccum = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    // // Repeat the partitioning with identity layouts
    // Tensor tOcOaccum = gmem_thr_copy_Oaccum.partition_S(cOaccum);
    // Tensor tOpOaccum = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
    // if (!Is_even_K) {
    //     #pragma unroll
    //     for (int k = 0; k < size(tOpOaccum); ++k) { tOpOaccum(k) = get<1>(tOcOaccum(0, 0, k)) < params.head_dims_q; }
    // }
    // // Load Oaccum in then scale and accumulate to O
    // for (int split = 0; split < kSplitKV; ++split) {
    //     flash::copy</*Is_even_MN=*/false, Is_even_K>(
    //         gmem_tiled_copy_Oaccum, tOgOaccum, tOrOaccum, tOcOaccum, tOpOaccum, params.head_num * params.seqlen_q - bidx * kBlockM
    //     );
    //     #pragma unroll
    //     for (int m = 0; m < size<1>(tOrOaccum); ++m) {
    //         int row = get<0>(tOcOaccum(0, m, 0));
    //         ElementAccum lse_scale = sLSE[split][row];
    //         #pragma unroll
    //         for (int k = 0; k < size<2>(tOrOaccum); ++k) {
    //             #pragma unroll
    //             for (int i = 0; i < size<0>(tOrOaccum); ++i) {
    //                 tOrO(i, m, k) += lse_scale * tOrOaccum(i, m, k);
    //             }
    //         }
    //     // if (cute::thread0()) { printf("lse_scale = %f, %f\n", sLSE[split][0], sLSE[split][1]); print(tOrOaccum); }
    //     }
    //     tOgOaccum.data() = tOgOaccum.data() + params.head_num * params.seqlen_q * params.head_dims_q;
    // }
    // // if (cute::thread0()) { print_tensor(tOrO); }

    // Tensor rO = flash::convert_type<Element>(tOrO);
    // // Write to gO
    // int o_batch_stride = params.head_num * params.seqlen_q * params.head_dims_q;
    // int o_head_stride = params.seqlen_q * params.head_dims_q;
    // int o_row_stride = params.head_dims_q;
    // #pragma unroll
    // for (int m = 0; m < size<1>(rO); ++m) {
    //     const int idx = bidx * kBlockM + get<0>(tOcOaccum(0, m, 0));
    //     if (idx < params.head_num * params.seqlen_q) {
    //         const int batch_idx = idx / (params.head_num * params.seqlen_q);
    //         const int head_idx = (idx - batch_idx * (params.head_num * params.seqlen_q)) / params.seqlen_q;
    //         // The index to the rows of Q
    //         const int row = idx - batch_idx * (params.head_num * params.seqlen_q) - head_idx * params.seqlen_q;
    //         auto o_ptr = reinterpret_cast<Element *>(params.output_ptr) + batch_idx * o_batch_stride
    //             + head_idx * o_head_stride + row * o_row_stride;
    //         #pragma unroll
    //         for (int k = 0; k < size<2>(rO); ++k) {
    //             if (Is_even_K || tOpOaccum(k)) {
    //                 const int col = get<1>(tOcOaccum(0, m, k));
    //                 Tensor gO = make_tensor(make_gmem_ptr(o_ptr + col),
    //                                         Shape<Int<decltype(size<0>(rO))::value>>{}, Stride<_1>{});
    //                 // TODO: Should check if this is using vectorized store, but it seems pretty fast
    //                 copy(rO(_, m, k), gO);
    //                 // if (bidx == 0 && tidx == 0) { printf("tidx = %d, idx = %d, batch_idx = %d, head_idx = %d, row = %d, col = %d\n", tidx, idx, batch_idx, head_idx, row, col); print(rO(_, m, k)); print(gO); }
    //                 // reinterpret_cast<uint64_t *>(o_ptr)[col / 4] = recast<uint64_t>(rO)(0, m, k);
    //             }
    //         }
    //     }
    // }
}