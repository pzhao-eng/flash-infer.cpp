#pragma once
#include <cute/tensor.hpp>
#include <cute/swizzle.hpp>
#include <cute/swizzle_layout.hpp>
#include <cute/int_tuple.hpp>
#include <cute/layout.hpp>
#include "include/softmax.h"
#include "include/utils.h"
#include "include/flash_infer_traits.h"
#include "include/flash_infer_params.h"
#include "include/flash_infer_kernel.h"

# define M_LOG2E	1.4426950408889634074	/* log_2 e */

void output_as_bin(const char *filename, const char *data, const int n);

template <typename kernel_traits>
void run_flash_infer(FlashInferParams params);


template <typename kernel_traits>
void run_flash_split_kv_infer(const FlashInferParams params);

template <typename Element, int SplitKV>
void prepare_input(FlashInferParams &params);