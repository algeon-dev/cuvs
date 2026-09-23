/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/c_api.h>

#include <rmm/error.hpp>

#include <exception>
#include <new>

namespace cuvs::core {

void set_last_error_kind(cuvsErrorKind_t kind);

/**
 * @brief Translates C++ exceptions into cuvs C-API error codes
 */
template <typename Fn>
cuvsError_t translate_exceptions(Fn func)
{
  cuvsError_t status;
  try {
    func();
    status = CUVS_SUCCESS;
    cuvsSetLastErrorText(NULL);
    set_last_error_kind(CUVS_ERROR_KIND_NONE);
  } catch (const rmm::bad_alloc& e) {
    cuvsSetLastErrorText(e.what());
    set_last_error_kind(CUVS_ERROR_KIND_DEVICE_OUT_OF_MEMORY);
    status = CUVS_ERROR;
  } catch (const std::bad_alloc& e) {
    cuvsSetLastErrorText(e.what());
    set_last_error_kind(CUVS_ERROR_KIND_HOST_OUT_OF_MEMORY);
    status = CUVS_ERROR;
  } catch (const std::exception& e) {
    cuvsSetLastErrorText(e.what());
    set_last_error_kind(CUVS_ERROR_KIND_UNKNOWN);
    status = CUVS_ERROR;
  } catch (...) {
    cuvsSetLastErrorText("unknown exception");
    set_last_error_kind(CUVS_ERROR_KIND_UNKNOWN);
    status = CUVS_ERROR;
  }
  return status;
}
}  // namespace cuvs::core
