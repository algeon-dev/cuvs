/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/math.hpp>
#include <raft/util/cuda_dev_essentials.cuh>  // DI, HDI

#include <cuda/std/limits>
#include <cuda_fp16.h>

namespace cuvs::distance::detail::ops {

/**
 * Whether an expanded L2 value `val = a_norm + b_norm - 2 <a, b>` lies within the rounding error
 * of a self pair, so that identical rows report an exact-zero distance.
 *
 * The norms and the dot product each sum `k` products, so standard dot-product error analysis
 * bounds the forward error of the expanded form by about k * u * (a_norm + b_norm), with u the
 * unit roundoff of AccT; eps = 2u doubles that bound. Products of half inputs are exact in a
 * float accumulator, so AccT's epsilon covers half data too. Only bit-equal norms qualify, and a
 * distinct pair above the bound keeps its distance even when both norms round to the same value.
 */
template <typename AccT>
HDI bool l2_exp_within_self_rounding(AccT a_norm, AccT b_norm, AccT val, AccT k)
{
  return a_norm == b_norm &&
         val <= k * cuda::std::numeric_limits<AccT>::epsilon() * (a_norm + b_norm);
}

// Epilogue operator for CUTLASS based kernel
template <typename DataT, typename AccT>
struct l2_exp_cutlass_op {
  bool sqrt;
  // Reduction dimension of the dot product, which scales the self-pair rounding bound.
  AccT k;

  HDI l2_exp_cutlass_op() noexcept : sqrt(false), k(0) {}
  HDI l2_exp_cutlass_op(bool isSqrt, AccT k_) noexcept : sqrt(isSqrt), k(k_) {}
  inline __device__ AccT operator()(AccT aNorm, AccT bNorm, AccT accVal) const noexcept
  {
    AccT outVal = aNorm + bNorm - AccT(2.0) * accVal;
    if (outVal < AccT(0) || l2_exp_within_self_rounding(aNorm, bNorm, outVal, k)) {
      outVal = AccT(0);
    }
    return sqrt ? raft::sqrt(outVal) : outVal;
  }

  __device__ AccT operator()(DataT aData) const noexcept
  {
    if constexpr (std::is_same_v<DataT, half> && std::is_same_v<AccT, float>) {
      return __half2float(aData);
    } else {
      return aData;
    }
  }
};

/**
 * @brief the expanded euclidean distance matrix calculation
 *
 * It computes the following equation:
 *
 * c_ij = - 2 sum_k x_ik * y_kj + ||x_i.||_2 + ||y_.j||_2
 *
 */
template <typename DataType, typename AccType, typename IdxType>
struct l2_exp_distance_op {
  using DataT = DataType;
  using AccT  = AccType;
  using IdxT  = IdxType;

  const bool sqrt;
  // Reduction dimension of the dot product, which scales the self-pair rounding bound.
  const IdxT k;

  l2_exp_distance_op(bool sqrt_, IdxT k_) noexcept : sqrt(sqrt_), k(k_) {}

  // Load norms of input data
  static constexpr bool use_norms = true;
  // Whether the core function requires so many instructions that it makes sense
  // to reduce loop unrolling, etc. We do this to keep compile times in check.
  static constexpr bool expensive_inner_loop = false;

  // Size of shared memory. This is normally decided by the kernel policy, but
  // some ops such as correlation_distance_op use more.
  template <typename Policy>
  static constexpr size_t shared_mem_size()
  {
    return Policy::SmemSize + ((Policy::Mblk + Policy::Nblk) * sizeof(AccT));
  }

  DI void core(AccT& acc, DataT& x, DataT& y) const
  {
    if constexpr ((std::is_same_v<AccT, float> && std::is_same_v<DataT, half>)) {
      acc += __half2float(x) * __half2float(y);
    } else {
      acc += x * y;
    }
  };

  template <typename Policy>
  DI void epilog(AccT acc[Policy::AccRowsPerTh][Policy::AccColsPerTh],
                 AccT* regxn,
                 AccT* regyn,
                 IdxT gridStrideX,
                 IdxT gridStrideY) const
  {
#pragma unroll
    for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = 0; j < Policy::AccColsPerTh; ++j) {
        AccT val = regxn[i] + regyn[j] - (AccT)2.0 * acc[i][j];
        if (val < AccT(0) || l2_exp_within_self_rounding(regxn[i], regyn[j], val, AccT(k))) {
          val = AccT(0);
        }
        acc[i][j] = val;
      }
    }
    if (sqrt) {
#pragma unroll
      for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
#pragma unroll
        for (int j = 0; j < Policy::AccColsPerTh; ++j) {
          acc[i][j] = raft::sqrt(acc[i][j]);
        }
      }
    }
  }

  constexpr l2_exp_cutlass_op<DataT, AccT> get_cutlass_op() const
  {
    return l2_exp_cutlass_op<DataT, AccT>(sqrt, static_cast<AccT>(k));
  }
};

}  // namespace cuvs::distance::detail::ops
