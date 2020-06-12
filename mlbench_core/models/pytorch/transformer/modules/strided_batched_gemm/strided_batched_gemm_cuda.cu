#include <vector>
#include <iostream>

#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>

#include "THC/THC.h"
#include "THC/THCBlas.h"

// symbol to be automatically resolved by PyTorch libs
extern THCState *state;


at::Tensor strided_batched_gemm_cuda(
    float beta,
    at::Tensor in_result,
    float alpha,
    at::Tensor batch1,
    at::Tensor batch2) {

  bool transpose_result;
  char transpose_batch1, transpose_batch2;
  int64_t lda, ldb, ldc;
  at::Tensor result, input1, input2;
  if (in_result.stride(1) == 1)
  {
    transpose_result = false;
    result = in_result;
    ldc = result.stride(2);
  }
  else if (in_result.stride(2) == 1)
  {
    transpose_result = true;

    at::Tensor swap = batch2;
    batch2 = batch1;
    batch1 = swap;

    result = in_result;
    ldc = result.stride(1);
  } else {
    AT_ASSERTM(false, "result should be contiguous");
  }

  if (batch1.stride(transpose_result ? 2 : 1) == 1 &&
      batch1.stride(transpose_result ? 1 : 2) != 0) {
    transpose_batch1 = 'n';
    input1 = batch1;
    lda = input1.stride(transpose_result ? 1 : 2);
  } else if (batch1.stride(transpose_result ? 1 : 2) == 1 &&
             batch1.stride(transpose_result ? 2 : 1) != 0) {
    transpose_batch1 = 't';
    input1 = batch1;
    lda = input1.stride(transpose_result ? 2 : 1);
  } else {
    AT_ASSERTM(false, "input1 should be contiguous");
  }

  if (batch2.stride(transpose_result ? 2 : 1) == 1 &&
      batch2.stride(transpose_result ? 1 : 2) != 0) {
    transpose_batch2 = 'n';
    input2 = batch2;
    ldb = input2.stride(transpose_result ? 1 : 2);
  } else if (batch2.stride(transpose_result ? 1 : 2) == 1 &&
             batch2.stride(transpose_result ? 2 : 1) != 0) {
    transpose_batch2 = 't';
    input2 = batch2;
    ldb = input2.stride(transpose_result ? 2 : 1);
  } else {
    AT_ASSERTM(false, "input2 should be contiguous");
  }
  int64_t num_batches = result.size(0);

  THCudaBlas_HgemmStridedBatched(
      state,
      transpose_batch1,
      transpose_batch2,
      result.size(transpose_result ? 2 : 1),
      result.size(transpose_result ? 1 : 2),
      input1.size(transpose_result ? 1 : 2),
      alpha,
      static_cast<const c10::Half *>(input1.data_ptr()), lda, input1.stride(0),
      static_cast<const c10::Half *>(input2.data_ptr()), ldb, input2.stride(0),
      beta,
      static_cast<c10::Half *>(result.data_ptr()), ldc, result.stride(0),
      num_batches);

  return in_result;
}


