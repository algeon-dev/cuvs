/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/core/c_api.h>
#include <dlpack/dlpack.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Select the k smallest values from a flat device array of n candidates.
 *
 * Treats `in_val` as a matrix of shape [1, n] and selects the `k` smallest
 * float values.  `out_idx` receives the int64 column positions of the selected
 * values in [0, n), so the caller can recover per-segment identity as:
 *
 *   segment_index        = out_idx[j] / segment_k
 *   position_in_segment  = out_idx[j] % segment_k
 *
 * @param[in]  res      cuvsResources_t handle
 * @param[in]  in_val   DLManagedTensor* shape [1, n], float32, device memory
 * @param[out] out_val  DLManagedTensor* shape [1, k], float32, device memory
 * @param[out] out_idx  DLManagedTensor* shape [1, k], int64,   device memory
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsSelectK(cuvsResources_t res,
                                    DLManagedTensor* in_val,
                                    DLManagedTensor* out_val,
                                    DLManagedTensor* out_idx);

/**
 * @brief Select the k smallest or largest values from each row of a device
 * matrix.
 *
 * Requires batch >= 1 and 1 <= k <= len. Each row of `in_val` has `len` values.
 * `out_idx` receives their int64 column positions in [0, len). When `sorted` is
 * true, each output row is ordered by
 * value (ascending for `select_min`, descending otherwise). The order of equal
 * values is unspecified.
 *
 * @param[in]  res         cuvsResources_t handle
 * @param[in]  in_val      DLManagedTensor* shape [batch, len], float32, row-major device
 * @param[out] out_val     DLManagedTensor* shape [batch, k], float32, row-major device
 * @param[out] out_idx     DLManagedTensor* shape [batch, k], int64, row-major device
 * @param[in]  select_min  Select smallest values if true, largest if false
 * @param[in]  sorted      Order each output row by value if true
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsSelectKBatched(cuvsResources_t res,
                                           DLManagedTensor* in_val,
                                           DLManagedTensor* out_val,
                                           DLManagedTensor* out_idx,
                                           bool select_min,
                                           bool sorted);

#ifdef __cplusplus
}
#endif
