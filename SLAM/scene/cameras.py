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
from torch import nn
import numpy as np
import math
from utils.graphics_utils import getWorld2View2, compute_se3_to_SE3

"""
Класс Camera_Pose является подклассом nn.Module из PyTorch и 
представляет собой модель, которая может быть использована для управления позой камеры в 3D пространстве
"""
class Camera_Pose(nn.Module):
    # trans: Вектор трансляции, по умолчанию [0.0, 0.0, 0.0]
    # scale: Масштабный коэффициент, по умолчанию 1.0
    def __init__(self,start_pose_w2c, FoVx, FoVy, image_width, image_height,
             trans=torch.tensor([0.0, 0.0, 0.0]), scale=1.0,
             ):
        # Вызов конструктора базового класса
        super(Camera_Pose, self).__init__()

        self.FoVx = FoVx
        self.FoVy = FoVy

        self.image_width = image_width
        self.image_height = image_height
        # Устанавливают дальнюю и ближнюю плоскости отсечения для камеры
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        self.cov_offset = 0
        """
        Параметры PyTorch nn.Parameter, инициализированные нормальным распределением с 
        малой дисперсией, хранят параметры вращения и трансляции соответственно. 
        Эти параметры будут обучаемыми, что означает, что PyTorch будет обновлять их во время обучения модели

        Инициализация параметров w и v нормальным распределением и перемещение их на то же устройство, где находится start_pose_w2c
        """
        self.w = nn.Parameter(torch.normal(0., 1e-6, size=(3,)).to(start_pose_w2c.device))
        self.v = nn.Parameter(torch.normal(0., 1e-6, size=(3,)).to(start_pose_w2c.device))
        """
        Метод forward использует параметры w и v для вычисления новой позы камеры, 
        затем вызывает метод update для обновления связанных с камерой параметров, 
        таких как матрицы вида и проекции. Функция compute_se3_to_SE3 преобразует векторы вращения и трансляции в гомогенную матрицу преобразования SE(3), 
        которая затем используется для обновления позы камеры
        """
        self.forward(start_pose_w2c)
    
    def forward(self, start_pose_w2c):
        """
        Преобразование параметров w и v в матрицу преобразования
        """
        deltaT=compute_se3_to_SE3(self.w,self.v)
        """
        Обновление позы камеры, вычисляется новая поза камеры в мировых координатах. Сначала инвертируется start_pose_w2c, затем умножается на deltaT, и результат снова инвертируется
        """
        self.pose_w2c = torch.matmul(deltaT, start_pose_w2c.inverse()).inverse()
        """
        Обновление матрицы трансформации и проекции камеры
        """
        self.update()
    
    def current_campose_c2w(self):
        return self.pose_w2c.inverse().clone().cpu().detach().numpy()
    
    def getProjectionMatrix(znear, zfar, fovX, fovY):
        """
        Вычисление значений для ограничивающих плоскостей проекции
        """
        tanHalfFovY = math.tan((fovY / 2))
        tanHalfFovX = math.tan((fovX / 2))

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        P = torch.zeros(4, 4)

        z_sign = 1.0 # если ось Z направлена к наблюдателю
        """
        Элементы матрицы заполняются в соответствии с формулами для матрицы проекции, которые основаны на параметрах проекции (znear, zfar, fovX, fovY):
        Элементы P[0,0] и P[1,1] определяют масштабирование по оси X и Y, соответственно.
        Элементы P[0,2] и P[1,2] определяют перспективное смещение.
        Элемент P[3,2] определяет знак глубины для проекции.
        Элементы P[2,2] и P[2,3] определяют масштабирование и смещение по оси Z.
        """
        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        return P
    

    def update(self):
        # Мировая матрица + перемещение на гпу
        self.world_view_transform = self.pose_w2c.transpose(0, 1).cuda()
        # Проекционная матрица
        self.projection_matrix = self.getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        # Вычисляется полная матрицу проекции, объединяя мировую матрицу вида и проекционную матрицу
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        # Вычисляетcя положение центра камеры в мировых координатах
        self.camera_center = self.world_view_transform.inverse()[3, :3]



class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = self.getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

