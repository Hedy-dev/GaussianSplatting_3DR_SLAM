#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from icomma_diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
"""
Функция render отвечает за рендеринг сцены с использованием модели гауссовских примитивов. Она принимает параметры камеры, 
модель гауссовских примитивов, параметры пайплайна рендеринга, цвет фона для настройки рендеринга.
"""
def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, compute_grad_cov2d=True):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    viewpoint_camera: объект, представляющий камеру.
    pc: объект класса GaussianModel, представляющий модель сцены.
    pipe: объект, содержащий параметры пайплайна рендеринга.
    bg_color: тензор, представляющий цвет фона.
    scaling_modifier (по умолчанию 1.0): множитель для масштабирования.
    override_color (по умолчанию None): цвет, который должен переопределить цвета модели.
    compute_grad_cov2d (по умолчанию True): флаг для вычисления градиентов ковариации в 2D.
    """
 
    # Создание тензора нулевых значений
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    # Вычисляются тангенсы половинных углов обзора по горизонтали и вертикали
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    # Создается объект GaussianRasterizationSettings, который содержит все необходимые параметры для растризации
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        compute_grad_cov2d=compute_grad_cov2d,
        proj_k=viewpoint_camera.projection_matrix
    )
    # Создание объекта растризатора
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    # Получение координат, прозрачности и ковариации гауссовских примитивов
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    #get_ply_file = pc.get_ply()
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    # Если необходимо, выполняется предварительное вычисление ковариаций в 3D
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # Если необходимо, вычисляются цвета из сферических гармоник (SH)
    # Определяются предварительно вычисленные цвета или переопределяются переданные цвета
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    print("camera_center: ", viewpoint_camera.camera_center)
    print("camera_pose: ", viewpoint_camera.world_view_transform)
    # Выполнение растризации 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        camera_center = viewpoint_camera.camera_center,
        camera_pose = viewpoint_camera.world_view_transform)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # return {"render": rendered_image,
    #         "viewspace_points": screenspace_points,
    #         "visibility_filter" : radii > 0,
    #         "radii": radii}
    return rendered_image