import torch
import numpy as np
import pyceres as ceres

from typing import Callable, Union, Tuple, List


def compute_jacobian_using_vmap(p: torch.Tensor, f: Callable) -> torch.Tensor:
    """
    Efficient Jacobian computation using functorch vmap

    :param p: initial value(s)
    :param f: function handle
    :return: Value of Jacobian f(p)
    """

    p.requires_grad_(True)
    y = f(p)
    I_N = torch.eye(y.shape[0]).cuda()

    def vjp(y_i):
        return torch.autograd.grad(y, p, y_i, retain_graph=True)

    jac = torch.vmap(vjp, in_dims=0)(I_N)[0]

    return jac


def compute_jacobian(p: torch.Tensor, f: Callable) -> torch.Tensor:
    jac = torch.autograd.functional.jacobian(f, p, strict=True, vectorize=False)
    return jac


def compute_hessian_using_vmap(p: torch.Tensor, f: Callable) -> torch.Tensor:
    """
    Efficient Hessian computation using functorch vmap

    :param p: initial value(s)
    :param f: function handle
    :return: Hessian matrix of f(p)
    """

    p.requires_grad_(True)
    y = f(p)
    grad_y = torch.autograd.grad(y, p, create_graph=True)[0]
    n = p.shape[0]

    def vjp(g_i):
        return torch.autograd.grad(grad_y, p, g_i, retain_graph=True)

    hessian = torch.vmap(vjp, in_dims=0)(torch.eye(n).cuda())

    return hessian


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
            jacobian = compute_jacobian(parameters_tensor, self.cost_function).flatten()
            jacobian_np_flat = jacobian.detach().cpu()
            jacobians[0][:] = jacobian_np_flat[:]

        torch.cuda.synchronize()

        return True


def levenberg_marquardt_ceres(p: torch.Tensor, cost_function: Callable, num_residuals: int) -> List[torch.Tensor]:
    options = ceres.SolverOptions()
    options.trust_region_strategy_type = ceres.TrustRegionStrategyType.LEVENBERG_MARQUARDT
    options.max_num_iterations = 10
    options.minimizer_progress_to_stdout = True

    problem = ceres.Problem()

    parameter_block_np = [p.numpy(force=True).astype(np.float64)]

    num_optimized_parameters = p.shape[0]
    epe_cost_function = EndPointErrorCostFunction(cost_function, num_residuals, num_optimized_parameters)
    loss = ceres.TrivialLoss()
    problem.add_residual_block(epe_cost_function, loss, parameter_block_np)

    summary = ceres.SolverSummary()
    ceres.solve(options, problem, summary)
    print(f"Ceres duration: {summary.total_time_in_seconds}")

    p_list = epe_cost_function.p_list
    p_list.append(torch.from_numpy(parameter_block_np[0]).cuda().to(p_list[0].dtype))

    return p_list


def lsq_lma_custom(
        p: torch.Tensor,
        function: Callable,
        jac_function: Callable = None,
        args: Union[Tuple, List] = (),
        ftol: float = 1e-8,
        ptol: float = 1e-8,
        gtol: float = 1e-8,
        tau: float = 1e-3,
        meth: str = 'lev',
        rho1: float = .25,
        rho2: float = .75,
        bet: float = 2,
        gam: float = 3,
        max_iter: int = 25) -> List[torch.Tensor]:
    """
    __author__ = "Christopher Hahne"
    __email__ = "inbox@christopherhahne.de"
    __license__ =
        Copyright (c) 2022 Christopher Hahne <inbox@christopherhahne.de>
        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.
        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.
        You should have received a copy of the GNU General Public License
        along with this program. If not, see <https://www.gnu.org/licenses/>.

    https://github.com/hahnec/torchimize

    Levenberg-Marquardt implementation for least-squares fitting of non-linear functions

    :param p: initial value(s)
    :param function: user-provided function which takes p (and additional arguments) as input
    :param jac_function: user-provided Jacobian function which takes p (and additional arguments) as input
    :param args: optional arguments passed to function
    :param ftol: relative change in cost function as stop condition
    :param ptol: relative change in independant variables as stop condition
    :param gtol: maximum gradient tolerance as stop condition
    :param tau: factor to initialize damping parameter
    :param meth: method which is default 'lev' for Levenberg and otherwise Marquardt
    :param rho1: first gain factor threshold for damping parameter adjustment for Marquardt
    :param rho2: second gain factor threshold for damping parameter adjustment for Marquardt
    :param bet: multiplier for damping parameter adjustment for Marquardt
    :param gam: divisor for damping parameter adjustment for Marquardt
    :param max_iter: maximum number of iterations
    :return: list of results
    """

    if len(args) > 0:
        # pass optional arguments to function
        fun = lambda p: function(p, *args)
    else:
        fun = function

    if jac_function is None:
        jac_fun = lambda p: compute_jacobian(p, f=fun)
    else:
        jac_fun = lambda p: jac_function(p, *args)

    f = fun(p)
    import time
    global_start_time = time.time()
    jac = jac_fun(p)
    g = torch.matmul(jac.T, f)
    H = torch.matmul(jac.T, jac)
    u = tau * torch.max(torch.diag(H))
    v = 2
    p_list = [p]
    it_idx = 0
    while len(p_list) < max_iter:
        time_start = time.process_time()
        # print(f"Iter {it_idx} begin", int(torch.cuda.memory_allocated() / 1024))
        D = torch.eye(jac.shape[1], device=jac.device)
        D *= 1 if meth == 'lev' else torch.max(torch.maximum(H.diagonal(), D.diagonal()))
        h = -torch.linalg.lstsq(H + u * D, g, rcond=None, driver=None)[0]
        f_h = fun(p + h)
        rho_denom = torch.matmul(h, u * h - g)
        rho_nom = torch.matmul(f, f) - torch.matmul(f_h, f_h)
        rho = rho_nom / rho_denom if rho_denom > 0 else torch.inf if rho_nom > 0 else -torch.inf
        if rho > 0:
            p = p + h
            jac = jac_fun(p)
            g = torch.matmul(jac.T, fun(p))
            H = torch.matmul(jac.T, jac)
        p_list.append(p.clone())
        f_prev = f.clone()
        f = fun(p)
        if meth == 'lev':
            u, v = (u * torch.max(torch.tensor([1 / 3, 1 - (2 * rho - 1) ** 3])), 2) if rho > 0 else (u * v, v * 2)
        else:
            u = u * bet if rho < rho1 else u / gam if rho > rho2 else u
        # stop conditions
        gcon = max(abs(g)) < gtol
        pcon = (h ** 2).sum() ** .5 < ptol * (ptol + (p ** 2).sum() ** .5)
        fcon = ((f_prev - f) ** 2).sum() < ((ftol * f) ** 2).sum() if rho > 0 else False
        # print(f"Iter {it_idx} end  ", int(torch.cuda.memory_allocated() / 1024))
        it_idx += 1
        print(f"Levenberg-Marquardt iteration {len(p_list)} took {time.process_time() - time_start}, rho {rho > 0}")
        if gcon or pcon or fcon:
            break

    print(f"Whole Levenberg-Marquardt Duration {time.time() - global_start_time}")
    return p_list
