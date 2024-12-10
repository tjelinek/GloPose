from typing import Callable

import torch
import pyceres as ceres

from optimization.optimization import compute_jacobian_using_vmap


class EndPointErrorCostFunction(ceres.CostFunction):

    def __init__(self, cost_function: Callable, num_residuals: int, num_optimized_parameters: int):
        super().__init__()

        self.num_residuals: int = num_residuals
        self.num_optimized_parameters: int = num_optimized_parameters
        self.cost_function: Callable = cost_function

        self.set_num_residuals(num_residuals)
        self.set_parameter_block_sizes([num_optimized_parameters])

        self.p_list = []

    def Evaluate(self, parameters, residuals, jacobians):
        parameters_tensor = torch.from_numpy(parameters[0]).to(torch.float).cuda()

        self.p_list.append(parameters_tensor.clone())

        function_eval = self.cost_function(parameters_tensor)
        function_eval_np = function_eval.numpy(force=True)
        residuals[:] = function_eval_np[:]

        if jacobians is not None:
            jacobian = compute_jacobian_using_vmap(parameters_tensor, self.cost_function).flatten()
            jacobian_np_flat = jacobian.detach().cpu()
            jacobians[0][:] = jacobian_np_flat[:]

        torch.cuda.synchronize()

        return True
