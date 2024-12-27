#include <cstdint>
#include <cstdlib>
#include <cuda.h>
#include <stdlib.h>
#include <cute/tensor.hpp>
#include <cute/swizzle.hpp>
#include <cute/swizzle_layout.hpp>
#include <cute/int_tuple.hpp>
#include <cute/layout.hpp>
#include "softmax.h"
#include "utils.h"
using namespace cute;

using Element = cutlass::half_t;
using ElementOut = cutlass::half_t;
using ElementAccum = float;
static constexpr int kNWarps = 4;
static constexpr int kNThreads = kNWarps * 32;
static constexpr int kBlockM = 128;
static constexpr int kBlockN = 128;

static constexpr int kHeadDim = 32;
// static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
static constexpr int kBlockKSmem = 32;
static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;
# define M_LOG2E	1.4426950408889634074	/* log_2 e */

using MMA_Atom_Arch = std::conditional_t<
    std::is_same_v<Element, cutlass::half_t>,
    MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
    MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;

using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, Element>;

using TiledMma = TiledMMA<
    MMA_Atom_Arch,
    Layout<Shape<Int<kNWarps>,_1,_1>>, // 4x1x1 or 8x1x1 thread group
    Tile<Int<16 * kNWarps>, _16, _16>>;

static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
// Using kBlockKSmem here is 6-10% faster than kBlockKGmem for d=128 because of bank conflicts.
// For example, for d=128, smem is split into 2 "pages", each page takes care of columns
// 0-63 and 64-127. If we have 16 threads per row for gmem read, when we write to smem,
// thread 0 - 7 will write to the first page and thread 8 - 15 will write to the second page,
// to the same banks.
static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                              Stride<Int<kGmemThreadsPerRow>, _1>>;

// We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
// from the same address by the same threadblock. This is slightly faster.
using Gmem_copy_struct = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
using GmemTiledCopyAB = decltype(
    make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                    GmemLayoutAtom{},
                    Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read

using SmemLayoutAtomQ = decltype(
    composition(Swizzle<kSwizzle, 3, 3>{},
                // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
                Layout<Shape<_8, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));

// using SmemLayoutAtomQ = decltype(
//                 // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
//                 Layout<Shape<_8, Int<kBlockKSmem>>,
//                         Stride<Int<kBlockKSmem>, _1>>{});

using SmemLayoutQ = decltype(tile_to_shape(
    SmemLayoutAtomQ{},
    Shape<Int<kBlockM>, Int<kHeadDim>>{}));

using SmemLayoutKV = decltype(tile_to_shape(
    SmemLayoutAtomQ{},
    Shape<Int<kBlockN>, Int<kHeadDim>>{}));

using SmemLayoutVtransposed = decltype(
    composition(SmemLayoutKV{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));


using SmemLayoutAtomO = decltype(
    composition(Swizzle<kSwizzle, 3, 3>{},
                Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
using SmemLayoutO = decltype(tile_to_shape(
    SmemLayoutAtomO{},
    Shape<Int<kBlockM>, Int<kHeadDim>>{}));
using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;
using SmemCopyAtomOaccum = Copy_Atom<DefaultCopy, ElementAccum>;

using GmemTiledCopyO = decltype(
    make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                    GmemLayoutAtom{},
                    Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store

static constexpr bool Share_Q_K_smem = false;
static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
static constexpr int kSmemSize = Share_Q_K_smem ? std::max(kSmemQSize, kSmemKVSize) : kSmemQSize + kSmemKVSize;

template<bool A_in_regs=false, bool B_in_regs=false, typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void gemm_warp(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                            Tensor4 const& tCsB, TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                            ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); }
    if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{})); }
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); }
            if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1)); }
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2>
__forceinline__ __device__ void copy_simplify(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                            const int max_MN = 0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    // There's no case where !Clear_OOB_K && Clear_OOB_MN
    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
      if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
        cute::copy(tiled_copy, S(_, m, _), D(_, m, _));
      } else if (Clear_OOB_MN) {
        cute::clear(D(_, m, _));
      }
    }
}

// A[n_row, num_heads, kHeadDim]
// B[n_col, num_heads, kHeadDim]
// V[n_col, num_heads, kHeadDim]
template <typename Element, typename ElementOut, bool is_even_MN>
__global__ void matrix_multiply_kernel(const Element * __restrict__ A, const Element * __restrict__ B, const Element * __restrict__ V,
    ElementOut *C, const int n_row, const int n_col, const int num_heads, const float softmax_scale, const float softmax_scale_log2,
    const int n_block) {
  int b_row = blockIdx.x;
  auto tidx = threadIdx.x;
  int head_idx = blockIdx.y;
  extern __shared__ Element smem[];

  Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(n_row, num_heads, Int<kHeadDim>{}), make_stride(num_heads * kHeadDim, Int<kHeadDim>{}, _1{}));
  Tensor gA = local_tile(mA(_, head_idx, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(b_row, 0));
  Tensor mB = make_tensor(make_gmem_ptr(B), make_shape(n_col, num_heads, Int<kHeadDim>{}), make_stride(num_heads * kHeadDim, Int<kHeadDim>{}, _1{}));
  Tensor gB = local_tile(mB(_, head_idx, _), Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, 0));
  Tensor mV = make_tensor(make_gmem_ptr(V), make_shape(n_col, num_heads, Int<kHeadDim>{}), make_stride(num_heads * kHeadDim, Int<kHeadDim>{}, _1{}));
  Tensor gV = local_tile(mV(_, head_idx, _), Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, 0));
  Tensor mO = make_tensor(make_gmem_ptr(C), make_shape(n_row, num_heads, Int<kHeadDim>{}), make_stride(num_heads * kHeadDim, Int<kHeadDim>{}, _1{}));
  Tensor gO = local_tile(mO(_, head_idx, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(b_row, 0));

  Tensor sA = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem)), SmemLayoutQ{});
  Tensor sB = make_tensor(make_smem_ptr(sA.data() + size(SmemLayoutQ{})), SmemLayoutKV{});
  Tensor sV = make_tensor(sB.data() + size(sB), SmemLayoutKV{});
  Tensor sVt = make_tensor(sV.data(), SmemLayoutVtransposed{});\
  Tensor sVtNoSwizzle = make_tensor(sV.data(), SmemLayoutVtransposedNoSwizzle{});

  // load data from global memory to shared memory
  GmemTiledCopyAB gmem_tiled_copy_AB;
  auto gemm_thr_copy_AB = gmem_tiled_copy_AB.get_thread_slice(tidx);
  Tensor tAgA = gemm_thr_copy_AB.partition_S(gA); // thread A copy of gA
  Tensor tAsA = gemm_thr_copy_AB.partition_D(sA);
  Tensor tBgB = gemm_thr_copy_AB.partition_S(gB);
  Tensor tBsB = gemm_thr_copy_AB.partition_D(sB);
  Tensor tVgV = gemm_thr_copy_AB.partition_S(gV);
  Tensor tVsV = gemm_thr_copy_AB.partition_D(sV);
  Tensor cQ = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA)));
  Tensor tQcQ = gemm_thr_copy_AB.partition_S(cQ);
  Tensor cKV = make_identity_tensor(make_shape(size<0>(sB), size<1>(sB)));
  Tensor tKVcKV = gemm_thr_copy_AB.partition_S(cKV);

  copy_simplify<is_even_MN>(gmem_tiled_copy_AB, tAgA, tAsA, tQcQ, n_row - b_row * kBlockM);
  copy_simplify<is_even_MN>(gmem_tiled_copy_AB, tBgB(_, _, _, n_block - 1), tBsB, tKVcKV, n_col);
  cute::cp_async_fence();
  copy_simplify<is_even_MN>(gmem_tiled_copy_AB, tVgV(_, _, _, n_block - 1), tVsV, tKVcKV, n_col);
  cute::cp_async_fence();

  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tidx);
  Tensor tSrA = thr_mma.partition_fragment_A(sA); // Source thread register A
  Tensor tSrB = thr_mma.partition_fragment_B(sB);
  Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);
  Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
  clear(acc_s);
  Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
  clear(acc_o);

  auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(tidx);
  Tensor tSsA = smem_thr_copy_A.partition_S(sA); // Source thread shared A

  auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(tidx);
  Tensor tSsB = smem_thr_copy_B.partition_S(sB);

  auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
  Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

  cute::cp_async_wait<1>();
  __syncthreads();

  gemm_warp(acc_s, tSrA, tSrB, tSsA, tSsB, tiled_mma, smem_tiled_copy_A, smem_tiled_copy_B, smem_thr_copy_A, smem_thr_copy_B);

  flash::Softmax<2 * size<1>(acc_o)> softmax;
  softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/false>(acc_s, acc_o, softmax_scale_log2);
  for (int i_b = n_block - 2; i_b >= 0; --i_b) {
    // prefetch next block of K
    copy_simplify<true>(gmem_tiled_copy_AB, tBgB(_, _, _, i_b), tBsB, tKVcKV);
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
    copy_simplify<true>(gmem_tiled_copy_AB, tVgV(_, _, _, i_b), tVsV, tKVcKV);
    cute::cp_async_fence();

    cute::cp_async_wait<1>();
    __syncthreads();
    clear(acc_s);
    gemm_warp(acc_s, tSrA, tSrB, tSsA, tSsB, tiled_mma, smem_tiled_copy_A, smem_tiled_copy_B, smem_thr_copy_A, smem_thr_copy_B);

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
  Tensor sO = make_tensor(sA.data(), SmemLayoutO{});    // (SMEM_M,SMEM_N)
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

  copy_simplify<is_even_MN>(gmem_tiled_copy_O, tOrO, tOgO, tOcO, n_row - b_row * kBlockM);
}

void output_as_bin(const char *filename, const char *data, const int n) {
  FILE *fp = fopen(filename, "wb");
  fwrite(data, sizeof(char), n, fp);
  fclose(fp);
}

void atten_cpu(const int n_row, const int n_col, const int n_heads, const int n_k,
  const Element *A_host, const Element *B_host, const Element *V_host, Element *C_host_ref) {
  float *acc_val_arr = (float*)malloc(n_col * sizeof(float));
  for (int i = 0; i < n_row; i++) {
    for (int i_head = 0; i_head < n_heads; i_head++) {
      float max_val = -INFINITY;
      for (int j = 0; j < n_col; j++) {
        float acc_val = 0.0;
        for (int k = 0; k < n_k; k++) {
          acc_val += A_host[i * n_heads * n_k + i_head * n_k + k] * B_host[k * n_heads * n_col + i_head * n_col + j];
        }
        acc_val /= sqrtf(float(n_k));
        max_val = std::max(max_val, acc_val);
        acc_val_arr[j] = acc_val;
      }

      float exp_sum = 0.0;
      for (int j = 0; j < n_col; j++) {
        acc_val_arr[j] = expf(acc_val_arr[j] - max_val);
        exp_sum += acc_val_arr[j];
      }

      for (int j = 0; j < n_col; j++) {
        acc_val_arr[j] /= exp_sum;
      }

      for (int i_k = 0; i_k < n_k; i_k++) {
        float val = 0.0;
        for (int j = 0; j < n_col; j++) {
          val += acc_val_arr[j] * float(V_host[j * n_heads * n_k + i_head * n_k + i_k]);
        }
        C_host_ref[i * n_heads * n_k + i_head * n_k + i_k] = ElementOut(val);
      }
    }
  }
  free(acc_val_arr);
}

int main() {

  const int n_row = 400;
  const int n_col = 32768;
  const int n_heads = 8;
  const int n_k = kHeadDim;
  const int n_test = 20;
  const int n_a = n_row * n_k * n_heads;
  const int n_b = n_k * n_col * n_heads;
  const int n_c = n_row * n_col * n_heads;
  const int n_bytes_a = n_a * sizeof(Element);
  const int n_bytes_b = n_b * sizeof(Element);
  const int n_bytes_v = n_b * sizeof(Element);
  const int n_bytes_c = n_a * sizeof(Element);
  Element *A_host = (Element*)malloc(n_bytes_a);
  Element *B_host = (Element*)malloc(n_bytes_b);
  Element *V_host = (Element*)malloc(n_bytes_v);
  Element *C_host = (Element*)malloc(n_bytes_c);
  Element *C_host_ref = (Element*)malloc(n_bytes_c);
  
  Element *A_device = nullptr;
  Element *B_device = nullptr;
  Element *V_device = nullptr;
  Element *C_device = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t start, stop;

  // set rand seed
  srand(0);
  for (int i = 0; i < n_a; i++) {
    // A_host[i] = static_cast<Element>(i + 1) * 0.1;
    // gen random float number
    float random = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    A_host[i] = static_cast<Element>(random);
    int index = rand() % n_a;
    A_host[index] = 0.0;
  }
  for (int i = 0; i < n_b; i++) {
    // B_host[i] = static_cast<Element>(i + 1) * 0.1;
    // V_host[i] = static_cast<Element>(i + 1) * 0.1;
    float random = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    B_host[i] = static_cast<Element>(random);
    V_host[i] = static_cast<Element>(random);
  }

  for (int i = 0; i < 100; i++) {
    int index = rand() % n_a;
    A_host[index] = 0.0;
    int index_b = rand() % n_b;
    B_host[index_b] = 0.0;
    V_host[index_b] = 0.0;
  }

  // output as bin file
  output_as_bin("q.bin", (const char*)A_host, n_bytes_a);
  output_as_bin("k.bin", (const char*)B_host, n_bytes_b);
  output_as_bin("v.bin", (const char*)V_host, n_bytes_v);

  printf("Input data sample: A[0]: %f B[0]: %f C[0]: %f\n", float(A_host[0]), float(B_host[0]), float(V_host[0]));

  cudaMalloc(&A_device, n_bytes_a);
  cudaMalloc(&B_device, n_bytes_b);
  cudaMalloc(&V_device, n_bytes_v);
  cudaMalloc(&C_device, n_bytes_c);
  cudaStreamCreate(&stream);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaMemcpy(A_device, A_host, n_bytes_a, cudaMemcpyHostToDevice);
  cudaMemcpy(B_device, B_host, n_bytes_b, cudaMemcpyHostToDevice);
  cudaMemcpy(V_device, V_host, n_bytes_v, cudaMemcpyHostToDevice);
  cudaMemset(C_device, 0, n_bytes_c);

  const int shared_memery_size = kBlockM * kHeadDim * sizeof(Element) + kBlockN * kHeadDim * sizeof(Element) * 2;
  assert(n_k % kHeadDim == 0);
  bool is_even_MN = n_row % kBlockM == 0 && n_col % kBlockN == 0;
  dim3 block(kNThreads);
  int grid_x = (n_row + kBlockM - 1) / kBlockM;
  dim3 grid(grid_x, n_heads);
  const int n_block = cute::ceil_div(n_col, kBlockN);
  float softmax_scale = 1.0 / sqrtf(float(n_k));
  float softmax_scale_log2 = softmax_scale * M_LOG2E;
  printf("\% [grid]: (%d, %d)\n", grid.x, grid.y);
  printf("\% [block]: (%d, %d)\n", block.x, block.y);
  printf("\% [shared memory]: %d : %d\n", shared_memery_size, kSmemSize);
  printf("\% [n_block]: %d\n", n_block, "softmax_scale: %f\n", softmax_scale);

  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(0);
  // print_latex(TiledMma{});
  // print_latex(MMA_Atom_Arch{});
  // print_latex(GmemTiledCopyAB{});
  // print(GmemLayoutAtom{});
  // print(GmemTiledCopyAB{});
  // print(SmemCopyAtom{});
  // auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
  // print_latex(smem_tiled_copy_A);
  // print(tiled_mma); printf("\n");

  if (is_even_MN) {
    matrix_multiply_kernel<Element, ElementOut, true><<<
      grid, block, shared_memery_size>>>(A_device, B_device, V_device, C_device, n_row, n_col, n_heads, softmax_scale, softmax_scale_log2, n_block);
  } else {
    matrix_multiply_kernel<Element, ElementOut, false><<<
      grid, block, shared_memery_size>>>(A_device, B_device, V_device, C_device, n_row, n_col, n_heads, softmax_scale, softmax_scale_log2, n_block);
  }

  cudaEventRecord(start, stream);
  for (int i = 0; i < n_test; i++) {
    if (is_even_MN) {
      matrix_multiply_kernel<Element, ElementOut, true><<<
        grid, block, kSmemSize, stream>>>(A_device, B_device, V_device, C_device, n_row, n_col, n_heads, softmax_scale, softmax_scale_log2, n_block);
    } else {
      matrix_multiply_kernel<Element, ElementOut, false><<<
        grid, block, kSmemSize, stream>>>(A_device, B_device, V_device, C_device, n_row, n_col, n_heads, softmax_scale, softmax_scale_log2, n_block);
    }
  }
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  // printf("Time: %f ms\n", ms / n_test);
  cudaMemcpy(C_host, C_device, n_bytes_c, cudaMemcpyDeviceToHost);

  bool is_same = true;
  // atten_cpu(n_row, n_col, n_heads, n_k, A_host, B_host, V_host, C_host_ref);
  // for (int i = 0; i < n_row && is_same; i++) {
  //   for (int i_head = 0; i_head < n_heads && is_same; i_head++) {
  //     for (int j = 0; j < n_k && is_same; j++) {
  //       if (fabs(float(C_host[i * n_heads * n_k + i_head * n_k + j]) - float(C_host_ref[i * n_heads * n_k + i_head * n_k + j]) > 0.05)) {
  //         printf("Error: i: %d, j: %d, C: %f, C_ref: %f\n", i, j, float(C_host[i * n_heads * n_k + i_head * n_k + j]), float(C_host_ref[i * n_heads * n_k + i_head * n_k + j]));
  //         is_same = false;
  //       }
  //     }
  //   }
  // }


  // for (int i = 0; i < n_k; i++) {
  //   int index = 126 * n_heads * n_k + 0 * n_k + i;
  //   int index_b = 124 * n_heads * n_k + 0 * n_k + i;
  //   printf("A[%d]: %f %f\n", i, float(A_host[index])), float(B_host[index_b]);
  // }
  std::string file_ref = "ref_val.bin";
  // read from bin file
  FILE *fp = fopen(file_ref.c_str(), "rb");
  if (fp == NULL) {
    printf("Error: open file %s failed\n", file_ref.c_str());
    return -1;
  }
  int ref_size = n_row * n_heads * n_k;
  float *ref_val = (float*)malloc(ref_size * sizeof(float));
  fread(ref_val, sizeof(float), ref_size, fp);
  fclose(fp);
  is_same == true;
  for (int i = 0; i < n_row && is_same; i++) {
    for (int i_head = 0; i_head < n_heads && is_same; i_head++) {
      for (int j = 0; j < n_k && is_same; j++) {
        int index = i * n_heads * n_k + i_head * n_k + j;
        if (fabs(float(C_host[index]) - ref_val[index]) > 0.01) {
          printf("Error: i: %d, j: %d, C: %f, C_ref: %f\n", i, j, float(C_host[index]), ref_val[index]);
          is_same = false;
        }
      }
    }
  }
  
  printf("\nshow val\n");
  printf("Q: %d K: %d V: %d Head: %d n_emb %d\n", n_row, n_col, n_k, n_heads, kHeadDim);
  printf("latency: %f ms\n", ms / n_test);
  printf("C_host[0] C: %f, C_ref: %f\n", float(C_host[0]), float(ref_val[0]));
  printf("C_host[13] C: %f, C_ref: %f\n", float(C_host[13]), float(ref_val[13]));

  if (is_same) {
    printf("Result is correct!\n");
  } else {
    printf("Result is wrong!\n");
  }

  // free
  free(A_host);
  free(B_host);
  free(C_host);
  cudaFree(A_device);
  cudaFree(B_device);
  cudaFree(C_device);
  cudaStreamDestroy(stream);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}