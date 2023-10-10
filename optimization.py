import time

from torchimize.functions.jacobian import jacobian_approx_t
from typing import Callable, Union, Tuple, List

import torch


# __author__ = "Christopher Hahne"
# __email__ = "inbox@christopherhahne.de"
# __license__ = """
#     Copyright (c) 2022 Christopher Hahne <inbox@christopherhahne.de>
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#     You should have received a copy of the GNU General Public License
#     along with this program. If not, see <https://www.gnu.org/licenses/>.
# """


def jacobian_approx_t_custom(p, f):
    # https://github.com/hahnec/torchimize
    """
    Numerical approximation for the multivariate Jacobian
    :param p: initial value(s)
    :param f: function handle
    :return: jacobian
    """

    # time_start = time.process_time()
    # try:
    #     jac = torch.autograd.functional.jacobian(f, p, vectorize=True)  # create_graph=True
    #     print(f"Vectorized Jacobian time {time.process_time() - time_start}")
    # except RuntimeError:
    #     print(f"Vectorized Jacobian time {time.process_time() - time_start}")
    # time_start = time.process_time()
    # fun = f
    jac = torch.autograd.functional.jacobian(f, p, strict=True, vectorize=False)
    # jac = torch.func.jacfwd(f)(p)
    # jac = torch.func.jacrev(f)(p)
    # breakpoint()
    # print(f"Non-vectorized Jacobian time {time.process_time() - time_start}")

    return jac


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
        max_iter: int = 25,
) -> List[torch.Tensor]:
    """
    # https://github.com/hahnec/torchimize
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
        # use numerical Jacobian if analytical is not provided
        jac_fun = lambda p: jacobian_approx_t_custom(p, f=fun)[:, 0, 0, :]
    else:
        jac_fun = lambda p: jac_function(p, *args)

    # print("Before function", int(torch.cuda.memory_allocated() / 1024))
    # breakpoint()
    f = fun(p)
    # print("Before Jacobian", int(torch.cuda.memory_allocated() / 1024))
    jac = jac_fun(p)
    # print("After  Jacobian", int(torch.cuda.memory_allocated() / 1024))
    # breakpoint()
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

    return p_list
