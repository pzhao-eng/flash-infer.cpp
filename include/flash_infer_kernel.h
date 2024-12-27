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
  cute::cp_async_wait<0>();
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