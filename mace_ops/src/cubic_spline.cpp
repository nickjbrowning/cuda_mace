#include "cubic_spline_impl.cuh"
#include "torch_utils.cuh"
#include "cubic_spline.h"

#include <torch/script.h>
#include <iostream>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;

torch::Tensor CubicSplineAutograd::forward(
        AutogradContext *ctx,
        torch::Tensor r,
        torch::Tensor coeffs,
        torch::Tensor r_width)
{
    auto result = evaluate_spline(
        r,
        coeffs,
        r_width);

    if (r.requires_grad())
    {
        ctx->save_for_backward({result[1]});
    }

    return result[0];
}

variable_list CubicSplineAutograd::backward(AutogradContext *ctx, variable_list grad_outputs)
{
    auto saved_variables = ctx->get_saved_variables();

    torch::Tensor R_deriv = saved_variables[0];
    
    torch::Tensor result = backward_spline(grad_outputs[0].contiguous(), R_deriv);
    
    torch::Tensor undef;

    return {result, undef, undef};
}

CubicSpline::CubicSpline(torch::Tensor r_basis, torch::Tensor R) {

    torch::Tensor r_width = torch::empty({1},
                                        torch::TensorOptions()
                                            .dtype(r_basis.dtype())
                                            .device(r_basis.device()));
    r_width[0] = r_basis[1] - r_basis[0];

    this->coeffs =  generate_coefficients(r_basis, R, r_width);
    
    this->r_width = r_width;
}

// wrapper class which we expose to the API.
torch::Tensor CubicSpline::forward(
        torch::Tensor r)
{
    return CubicSplineAutograd::apply(r, this->coeffs, this->r_width);
}

torch::Tensor CubicSpline::get_coefficients() {
    return this->coeffs;
}

TORCH_LIBRARY(cubic_spline, m)
{
    m.class_<CubicSpline>("CubicSpline")
        .def(torch::init<torch::Tensor, torch::Tensor>())

        .def("forward", &CubicSpline::forward)
        .def("get_coefficients", &CubicSpline::get_coefficients)
        .def_pickle(
            [](const c10::intrusive_ptr<CubicSpline> &self) -> std::vector<torch::Tensor>
            {
                return self->__getstate__();
            },
            [](const std::vector<torch::Tensor> &state) -> c10::intrusive_ptr<CubicSpline>
            {
                auto obj = c10::make_intrusive<CubicSpline>();
                obj->__setstate__(state);
                return obj;
            });
        
}
