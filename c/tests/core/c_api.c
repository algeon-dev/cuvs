/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/c_api.h>
#include <cuvs/version_config.h>

#include <stdio.h>
#include <stdlib.h>

int main()
{
  // Create resources
  cuvsResources_t res;
  cuvsError_t create_error = cuvsResourcesCreate(&res);
  if (create_error == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Set CUDA stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cuvsError_t stream_error = cuvsStreamSet(res, stream);
  if (stream_error == CUVS_ERROR) { exit(EXIT_FAILURE); }

  size_t pool_size = 1;
  if (cuvsResourcesGetStreamPoolSize(res, &pool_size) != CUVS_SUCCESS || pool_size != 0) {
    exit(EXIT_FAILURE);
  }
  if (cuvsResourcesSetStreamPool(res, 0) != CUVS_ERROR ||
      cuvsGetLastErrorKind() != CUVS_ERROR_KIND_UNKNOWN || cuvsGetLastErrorText() == NULL) {
    exit(EXIT_FAILURE);
  }
  if (cuvsResourcesSetStreamPool(res, 2) != CUVS_SUCCESS ||
      cuvsGetLastErrorKind() != CUVS_ERROR_KIND_NONE || cuvsGetLastErrorText() != NULL ||
      cuvsResourcesGetStreamPoolSize(res, &pool_size) != CUVS_SUCCESS || pool_size != 2) {
    exit(EXIT_FAILURE);
  }
  cudaStream_t pool_streams[2];
  if (cuvsResourcesGetStreamPoolStream(res, 0, &pool_streams[0]) != CUVS_SUCCESS ||
      cuvsResourcesGetStreamPoolStream(res, 1, &pool_streams[1]) != CUVS_SUCCESS ||
      pool_streams[0] == NULL || pool_streams[1] == NULL || pool_streams[0] == pool_streams[1] ||
      pool_streams[0] == stream || pool_streams[1] == stream ||
      cuvsResourcesGetStreamPoolStream(res, 2, &pool_streams[0]) != CUVS_ERROR) {
    exit(EXIT_FAILURE);
  }

  // Allocate memory
  void* ptr;
  size_t bytes      = 1024;
  cuvsError_t error = cuvsRMMAlloc(res, &ptr, bytes);
  if (error == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Free memory
  error = cuvsRMMFree(res, ptr, bytes);
  if (error == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Enable pool memory resource
  error = cuvsRMMPoolMemoryResourceEnable(10, 100, false);
  if (error == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Allocate memory again
  error = cuvsRMMAlloc(res, &ptr, 1024);
  if (error == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Free memory
  error = cuvsRMMFree(res, ptr, 1024);
  if (error == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Reset pool memory resource
  error = cuvsRMMMemoryResourceReset();
  if (error == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Enable pool memory resource (managed)
  error = cuvsRMMPoolMemoryResourceEnable(10, 100, true);
  if (error == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Allocate memory again
  error = cuvsRMMAlloc(res, &ptr, 1024);
  if (error == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Free memory
  error = cuvsRMMFree(res, ptr, 1024);
  if (error == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Reset pool memory resource
  error = cuvsRMMMemoryResourceReset();
  if (error == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Enable async memory resource
  error = cuvsRMMAsyncMemoryResourceEnable();
  if (error == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Allocate memory again
  error = cuvsRMMAlloc(res, &ptr, 1024);
  if (error == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Free memory
  error = cuvsRMMFree(res, ptr, 1024);
  if (error == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Reset async memory resource
  error = cuvsRMMMemoryResourceReset();
  if (error == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Alloc memory on host (pinned)
  void* ptr3;
  cuvsError_t alloc_error_pinned = cuvsRMMHostAlloc(&ptr3, 1024);
  if (alloc_error_pinned == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Free memory
  cuvsError_t free_error_pinned = cuvsRMMHostFree(ptr3, 1024);
  if (free_error_pinned == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Destroy resources
  error = cuvsResourcesDestroy(res);
  if (error == CUVS_ERROR) { exit(EXIT_FAILURE); }

  // Check version
  uint16_t major, minor, patch;
  error = cuvsVersionGet(&major, &minor, &patch);
  if (error == CUVS_ERROR || major != CUVS_VERSION_MAJOR || minor != CUVS_VERSION_MINOR ||
      patch != CUVS_VERSION_PATCH) {
    exit(EXIT_FAILURE);
  }

  return 0;
}
