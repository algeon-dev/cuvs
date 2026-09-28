/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "distance_base.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <vector>

namespace cuvs {
namespace distance {

template <typename DataType, typename OutputType = DataType>
class DistanceEucExpTest
  : public DistanceTest<cuvs::distance::DistanceType::L2Expanded, DataType, OutputType> {};

template <typename DataType, typename OutputType = DataType>
class DistanceEucExpTestXequalY
  : public DistanceTestSameBuffer<cuvs::distance::DistanceType::L2Expanded, DataType, OutputType> {
};

const std::vector<DistanceInputs<float>> inputsf = {
  {0.001f, 128, (65536 + 128) * 128, 8, true, 1234ULL},
  {0.001f, 2048, 4096, 128, true, 1234ULL},
  {0.001f, 1024, 1024, 32, true, 1234ULL},
  {0.001f, 1024, 32, 1024, true, 1234ULL},
  {0.001f, 32, 1024, 1024, true, 1234ULL},
  {0.003f, 1024, 1024, 1024, true, 1234ULL},
  {0.003f, 1021, 1021, 1021, true, 1234ULL},
  {0.001f, (65536 + 128) * 128, 128, 8, false, 1234ULL},
  {0.001f, 1024, 1024, 32, false, 1234ULL},
  {0.001f, 1024, 32, 1024, false, 1234ULL},
  {0.001f, 32, 1024, 1024, false, 1234ULL},
  {0.003f, 1024, 1024, 1024, false, 1234ULL},
  {0.003f, 1021, 1021, 1021, false, 1234ULL},
};

const std::vector<DistanceInputs<float>> inputsXeqYf = {
  {0.01f, 2048, 4096, 128, true, 1234ULL},
  {0.01f, 1024, 1024, 32, true, 1234ULL},
  {0.01f, 1024, 32, 1024, true, 1234ULL},
  {0.01f, 32, 1024, 1024, true, 1234ULL},
  {0.03f, 1024, 1024, 1024, true, 1234ULL},
  {0.03f, 1021, 1021, 1021, true, 1234ULL},
  {0.01f, 1024, 1024, 32, false, 1234ULL},
  {0.01f, 1024, 32, 1024, false, 1234ULL},
  {0.01f, 32, 1024, 1024, false, 1234ULL},
  {0.03f, 1024, 1024, 1024, false, 1234ULL},
  {0.03f, 1021, 1021, 1021, false, 1234ULL},
};

const std::vector<DistanceInputs<half, float>> inputsh = {
  {0.001f, 128, (65536 + 128) * 128, 8, true, 1234ULL},
  {0.001f, 2048, 4096, 128, true, 1234ULL},
  {0.001f, 1024, 1024, 32, true, 1234ULL},
  {0.001f, 1024, 32, 1024, true, 1234ULL},
  {0.001f, 32, 1024, 1024, true, 1234ULL},
  {0.003f, 1024, 1024, 1024, true, 1234ULL},
  {0.003f, 1021, 1021, 1021, true, 1234ULL},
  {0.001f, (65536 + 128) * 128, 128, 8, false, 1234ULL},
  {0.001f, 1024, 1024, 32, false, 1234ULL},
  {0.001f, 1024, 32, 1024, false, 1234ULL},
  {0.001f, 32, 1024, 1024, false, 1234ULL},
  {0.003f, 1024, 1024, 1024, false, 1234ULL},
  {0.003f, 1021, 1021, 1021, false, 1234ULL},
};

const std::vector<DistanceInputs<half, float>> inputsXeqYh = {
  {0.01f, 2048, 4096, 128, true, 1234ULL},
  {0.01f, 1024, 1024, 32, true, 1234ULL},
  {0.01f, 1024, 32, 1024, true, 1234ULL},
  {0.01f, 32, 1024, 1024, true, 1234ULL},
  {0.03f, 1024, 1024, 1024, true, 1234ULL},
  {0.03f, 1021, 1021, 1021, true, 1234ULL},
  {0.01f, 1024, 1024, 32, false, 1234ULL},
  {0.01f, 1024, 32, 1024, false, 1234ULL},
  {0.01f, 32, 1024, 1024, false, 1234ULL},
  {0.03f, 1024, 1024, 1024, false, 1234ULL},
  {0.03f, 1021, 1021, 1021, false, 1234ULL},
};

typedef DistanceEucExpTest<float> DistanceEucExpTestF;
TEST_P(DistanceEucExpTestF, Result)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(devArrMatch(
    dist_ref.data(), dist.data(), m, n, cuvs::CompareApprox<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(DistanceTests, DistanceEucExpTestF, ::testing::ValuesIn(inputsf));

typedef DistanceEucExpTest<half, float> DistanceEucExpTestH;
TEST_P(DistanceEucExpTestH, Result)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(devArrMatch(
    dist_ref.data(), dist.data(), m, n, cuvs::CompareApprox<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(DistanceTests, DistanceEucExpTestH, ::testing::ValuesIn(inputsh));

typedef DistanceEucExpTestXequalY<float> DistanceEucExpTestXequalYF;
TEST_P(DistanceEucExpTestXequalYF, Result)
{
  int m = params.m;
  ASSERT_TRUE(cuvs::devArrMatch(dist_ref[0].data(),
                                dist[0].data(),
                                m,
                                m,
                                cuvs::CompareApprox<float>(params.tolerance),
                                stream));
  ASSERT_TRUE(cuvs::devArrMatch(dist_ref[1].data(),
                                dist[1].data(),
                                m / 2,
                                m,
                                cuvs::CompareApprox<float>(params.tolerance),
                                stream));
}
INSTANTIATE_TEST_CASE_P(DistanceTests,
                        DistanceEucExpTestXequalYF,
                        ::testing::ValuesIn(inputsXeqYf));

typedef DistanceEucExpTestXequalY<half, float> DistanceEucExpTestXequalYH;
TEST_P(DistanceEucExpTestXequalYH, Result)
{
  int m = params.m;
  ASSERT_TRUE(cuvs::devArrMatch(dist_ref[0].data(),
                                dist[0].data(),
                                m,
                                m,
                                cuvs::CompareApprox<float>(params.tolerance),
                                stream));
  ASSERT_TRUE(cuvs::devArrMatch(dist_ref[1].data(),
                                dist[1].data(),
                                m / 2,
                                m,
                                cuvs::CompareApprox<float>(params.tolerance),
                                stream));
}
INSTANTIATE_TEST_CASE_P(DistanceTests,
                        DistanceEucExpTestXequalYH,
                        ::testing::ValuesIn(inputsXeqYh));

const std::vector<DistanceInputs<double>> inputsd = {
  {0.001, 1024, 1024, 32, true, 1234ULL},
  {0.001, 1024, 32, 1024, true, 1234ULL},
  {0.001, 32, 1024, 1024, true, 1234ULL},
  {0.003, 1024, 1024, 1024, true, 1234ULL},
  {0.001, 1024, 1024, 32, false, 1234ULL},
  {0.001, 1024, 32, 1024, false, 1234ULL},
  {0.001, 32, 1024, 1024, false, 1234ULL},
  {0.003, 1024, 1024, 1024, false, 1234ULL},
};
typedef DistanceEucExpTest<double> DistanceEucExpTestD;
TEST_P(DistanceEucExpTestD, Result)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(devArrMatch(
    dist_ref.data(), dist.data(), m, n, cuvs::CompareApprox<double>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(DistanceTests, DistanceEucExpTestD, ::testing::ValuesIn(inputsd));

class BigMatrixEucExp : public BigMatrixDistanceTest<cuvs::distance::DistanceType::L2Expanded> {};
TEST_F(BigMatrixEucExp, Result) {}

// Both x rows have a float32 squared norm of exactly 1.0f, the same as y, so the
// equal-norm self-distance guard applies to the near-unit pair even though its true
// squared distance (about 3.99987e-4) is far above float rounding noise.
TEST(DistanceEucExpNearUnit, EqualNormPairKeepsDistance)
{
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  std::vector<float> x_h = {0.9998000066665778f, 0.01999866669333308f, 1.0f, 0.0f};
  std::vector<float> y_h = {1.0f, 0.0f};
  auto x                 = raft::make_device_matrix<float, std::int64_t>(handle, 2, 2);
  auto y                 = raft::make_device_matrix<float, std::int64_t>(handle, 1, 2);
  auto dist              = raft::make_device_matrix<float, std::int64_t>(handle, 2, 1);
  raft::update_device(x.data_handle(), x_h.data(), x_h.size(), stream);
  raft::update_device(y.data_handle(), y_h.data(), y_h.size(), stream);

  cuvs::distance::pairwise_distance(handle,
                                    raft::make_const_mdspan(x.view()),
                                    raft::make_const_mdspan(y.view()),
                                    dist.view(),
                                    cuvs::distance::DistanceType::L2Expanded);

  std::vector<float> dist_h(2);
  raft::update_host(dist_h.data(), dist.data_handle(), dist_h.size(), stream);
  raft::resource::sync_stream(handle);

  EXPECT_NEAR(dist_h[0], 3.99987e-4f, 1e-6f);
  EXPECT_EQ(dist_h[1], 0.0f);
}
}  // end namespace distance
}  // namespace cuvs
