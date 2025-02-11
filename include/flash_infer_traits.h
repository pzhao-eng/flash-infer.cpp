
// ref https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/kernel_traits.h
#pragma once

#include <cute/tensor.hpp>
#include <cute/swizzle.hpp>
#include <cute/swizzle_layout.hpp>
#include <cute/int_tuple.hpp>
#include <cute/layout.hpp>

using namespace cute;
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::half_t,
         int kSPlitKV_ = 1>
struct FlashInferTraits {

  using Element = elem_type;
  static constexpr int kNWarps = 4;
  static constexpr int kNThreads = kNWarps * 32;
  static constexpr int kBlockM = kBlockM_;
  static constexpr int kBlockN = kBlockN_;
  static constexpr int kSplitKV = kSPlitKV_;

  static constexpr int kHeadDim = kHeadDim_;
  static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
  static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

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
  using GmemTiledCopyQKV = decltype(
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

  using GmemTiledCopyO = decltype(
      make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                      GmemLayoutAtom{},
                      Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store

  static constexpr bool Share_Q_K_smem = false;
  static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
  static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
  static constexpr int kSmemSize = Share_Q_K_smem ? std::max(kSmemQSize, kSmemKVSize) : kSmemQSize + kSmemKVSize;
};