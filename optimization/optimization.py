import torch
import numpy as np
import pyceres as ceres

from typing import Callable, Union, Tuple, List

from models.flow_loss_model import LossFunctionWrapper
from optimization.epe import EndPointErrorCostFunction


def run_levenberg_marquardt_method(
        flow_observations,
        flow_frames,
        keyframes,
        flow_arcs,
        encoder,
        rendering,
        loss_function,
        image_shape,
        config,
        use_custom_jacobian=False,
        levenberg_marquardt_implementation='ceres',
        max_iterations=100
):
    """
    Standalone implementation of the Levenberg-Marquardt method.

    Args:
        observations: Observed images and segmentations.
        flow_observations: Observed flows, segmentations, occlusions, uncertainties.
        flow_frames: List of flow frames.
        keyframes: List of keyframes.
        flow_arcs: Pairs of flow-to-keyframe relationships.
        encoder: Encoder object for inference.
        rendering: Rendering object for flow rendering and image generation.
        loss_function: Loss function wrapper for calculating optimization losses.
        image_shape: Shape of the image (width and height).
        config: Configuration object with parameters for optimization.
        use_custom_jacobian (bool): Flag to use custom Jacobian.
        levenberg_marquardt_implementation (str): 'ceres' or 'custom'.
        max_iterations (int): Maximum number of iterations for optimization.

    Returns:
        dict: Best model information including losses and encoder state.
    """
    # Extract and zero-out other loss coefficients
    loss_coefs_names = [
        'loss_laplacian_weight', 'loss_tv_weight', 'loss_iou_weight', 'loss_dist_weight',
        'loss_q_weight', 'loss_t_weight', 'loss_rgb_weight', 'loss_flow_weight'
    ]

    for field_name in loss_coefs_names:
        if field_name != "loss_flow_weight":
            setattr(config, field_name, 0)

    observed_flows = flow_observations.observed_flow
    observed_flows_segmentations = flow_observations.observed_flow_segmentation

    flow_arcs_indices = [(flow_frames.index(pair[0]), keyframes.index(pair[1])) for pair in flow_arcs]

    # Perform encoder inference
    encoder_result, encoder_result_flow_frames = encoder.frames_and_flow_frames_inference(keyframes, flow_frames)
    kf_translations = encoder_result.translations[0].detach()
    kf_quaternions = encoder_result.quaternions.detach()
    trans_quats = torch.cat([kf_translations, kf_quaternions], dim=-1).squeeze().flatten()

    # Create the loss model
    flow_loss_model = LossFunctionWrapper(
        encoder_result, encoder_result_flow_frames, encoder, rendering, flow_arcs_indices,
        loss_function, observed_flows, observed_flows_segmentations,
        rendering.width, rendering.height, image_shape.width, image_shape.height
    )

    fun = flow_loss_model.forward
    jac_function = None
    if use_custom_jacobian:
        jac_function = flow_loss_model.compute_jacobian

    # Select implementation
    if levenberg_marquardt_implementation == 'ceres':
        coefficients_list = levenberg_marquardt_ceres(
            p=trans_quats, cost_function=fun,
            num_residuals=config.flow_sgd_n_samples * len(flow_arcs)
        )
    elif levenberg_marquardt_implementation == 'custom':
        coefficients_list = lsq_lma_custom(
            p=trans_quats, function=fun, args=(), jac_function=jac_function, max_iter=max_iterations
        )
    else:
        raise ValueError("'levenberg_marquardt_implementation' must be either 'custom' or 'ceres'")

    translations = []
    quaternions = []
    # Optimization loop
    for epoch, trans_quats in enumerate(coefficients_list):
        trans_quats = trans_quats.unflatten(-1, (1, trans_quats.shape[-1] // 7, 7))

        row_translation = trans_quats[:, :3]
        row_quaternion = trans_quats[:, 3:]

        translations.append(row_translation)
        translations.append(row_quaternion)

    translations = torch.cat(translations, dim=0)
    quaternions = torch.cat(quaternions, dim=0)

    return translations, quaternions


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
        jac_fun = lambda p: compute_jacobian_using_vmap(p, f=fun)
        # jac_fun = lambda p: compute_jacobian(p, f=fun)
    else:
        jac_fun = lambda p: jac_function(p, *args)

    f = fun(p)
    import time
    torch.cuda.synchronize()
    global_start_time = time.time()
    jac = jac_fun(p)
    g = torch.matmul(jac.T, f)
    H = torch.matmul(jac.T, jac)
    u = tau * torch.max(torch.diag(H))
    v = 2
    p_list = [p]
    it_idx = 0

    print(f"Jacobian size: {jac.shape}")

    while len(p_list) < max_iter:
        torch.cuda.synchronize()
        time_start = time.process_time()

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

        it_idx += 1
        torch.cuda.synchronize()
        print(f"Levenberg-Marquardt iteration {len(p_list)} took {time.process_time() - time_start}, rho {rho > 0}")
        if gcon or pcon or fcon:
            break

    torch.cuda.synchronize()
    print(f"Whole Levenberg-Marquardt Duration {time.time() - global_start_time}")
    return p_list
