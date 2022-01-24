#include <tuple>
#include <string>
#include <paddle/extension.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>

#include "core.h"

#define CHECK_INPUT(x) \
    PD_CHECK(x.place() == paddle::PlaceType::kGPU, \
             #x " must be a GPU Tensor.")

#define CHECK_FLOAT(x) \
    PD_CHECK(x.type() == paddle::DataType::FLOAT32, \
             #x " must be a Float tensor")

#define CHECK_INT(x)                                            \
    PD_CHECK(x.type() == paddle::DataType::INT32 || x.type() == paddle::DataType::INT64,   \
              #x " must be a Int tensor")

#define DIM(x) x.shape().size()

std::vector<paddle::Tensor> rnnt_loss(const paddle::Tensor& xs, const paddle::Tensor& ys,
                                      const paddle::Tensor& xn, const paddle::Tensor& yn,
                                      const int blank = 0, const float fastemit_lambda = 0.0) {
    // Check device
    CHECK_INPUT(xs);
    CHECK_INPUT(ys);
    CHECK_INPUT(xn);
    CHECK_INPUT(yn);
    // Check types
    CHECK_FLOAT(xs);
    CHECK_INT(ys);
    CHECK_INT(xn);
    CHECK_INT(yn);
    // Check number of dimensions and elements
    PD_CHECK(DIM(xs) == 4, "xs must have 4 dimensions");
    PD_CHECK(xn.size() == xs.shape()[0], "xn shape must be equal (N,)");
    PD_CHECK(yn.size() == xs.shape()[0], "yn shape must be equal (N,)");
    PD_CHECK(xs.shape()[2] == ys.shape()[1] + 1, "ys shape (N, U-1) mismatched with xs (N, T, U, V)");

    const auto N = xs.shape()[0];
    const auto T = xs.shape()[1];
    const auto U = xs.shape()[2];
    const auto V = xs.shape()[3];

    paddle::Tensor grads = paddle::Tensor(paddle::PlaceType::kGPU, xs.shape());
    paddle::Tensor costs = paddle::Tensor(paddle::PlaceType::kGPU, std::vector<int64_t> {N});
    paddle::Tensor counts = paddle::Tensor(paddle::PlaceType::kGPU, std::vector<int64_t> {N, U*2});
    paddle::Tensor alphas = paddle::Tensor(paddle::PlaceType::kGPU, std::vector<int64_t> {N, T, U});
    paddle::Tensor betas = paddle::Tensor(paddle::PlaceType::kGPU, std::vector<int64_t> {N, T, U});

    auto stream = xs.stream();

    rnntStatus_t status;

    if (blank == -1) {

        PD_CHECK(V == 2, "xs must have values only for blank and label");

        status = run_warp_rnnt_gather(stream,
                                      (unsigned int *)counts.mutable_data<int>(xs.place()),
                                      alphas.mutable_data<float>(xs.place()), 
                                      betas.mutable_data<float>(xs.place()),
                                      xs.data<float>(),
                                      grads.mutable_data<float>(xs.place()), 
                                      costs.mutable_data<float>(xs.place()),
                                      xn.data<int>(), 
                                      yn.data<int>(),
                                      N, T, U, fastemit_lambda
        );

    } else {

        status = run_warp_rnnt(stream,
                               (unsigned int *)counts.mutable_data<int>(xs.place()),
                               alphas.mutable_data<float>(xs.place()), 
                               betas.mutable_data<float>(xs.place()),
                               ys.data<int>(), 
                               xs.data<float>(),
                               grads.mutable_data<float>(xs.place()), 
                               costs.mutable_data<float>(xs.place()),
                               xn.data<int>(), 
                               yn.data<int>(),
                               N, T, U, V, blank, fastemit_lambda
        );
    }

    PD_CHECK(status == RNNT_STATUS_SUCCESS, "rnnt_loss status " + std::to_string(status));

    return {costs, grads};
}

std::vector<std::vector<int64_t>> RnntLossInferShape(
    std::vector<int64_t> xs_shape,
    std::vector<int64_t> ys_shape,
    std::vector<int64_t> xn_shape,
    std::vector<int64_t> yn_shape) {
    return {xs_shape, std::vector<int64_t>{xs_shape[0]}};
}

std::vector<paddle::DataType> RnntLossInferDtype(
    paddle::DataType xs_dtype,
    paddle::DataType ys_dtype,
    paddle::DataType xn_dtype,
    paddle::DataType yn_dtype) {
    return {xs_dtype, xs_dtype};
}

PD_BUILD_OP(rnnt_loss)
    .Inputs({"xs", "ys", "xn", "yn"})
    .Outputs({"costs", "grads"})
    .Attrs({"blank: int",
            "fastemit_lambda: float"})
    .SetKernelFn(PD_KERNEL(rnnt_loss))
    .SetInferShapeFn(PD_INFER_SHAPE(RnntLossInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(RnntLossInferDtype));