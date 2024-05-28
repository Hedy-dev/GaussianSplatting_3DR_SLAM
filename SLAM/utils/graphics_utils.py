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
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View_torch(R, t):
    Rt = torch.zeros(4, 4)
    Rt[:3, :3] = R.t()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

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

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def skew_symmetric(w):
    w0,w1,w2 = w.unbind(dim=-1)
    O = torch.zeros_like(w0)
    wx = torch.stack([torch.stack([O,-w2,w1],dim=-1),
                        torch.stack([w2,O,-w0],dim=-1),
                        torch.stack([-w1,w0,O],dim=-1)],dim=-2)
    return wx
def taylor_A(x,nth=10):
        # Taylor expansion of sin(x)/x
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            if i>0: denom *= (2*i)*(2*i+1)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans
  
def taylor_B(x,nth=10):
    # Taylor expansion of (1-cos(x))/x**2
    ans = torch.zeros_like(x)
    denom = 1.
    for i in range(nth+1):
        denom *= (2*i+1)*(2*i+2)
        ans = ans+(-1)**i*x**(2*i)/denom
    return ans
def taylor_C(x,nth=10):
    # Taylor expansion of (x-sin(x))/x**3
    ans = torch.zeros_like(x)
    denom = 1.
    for i in range(nth+1):
        denom *= (2*i+2)*(2*i+3)
        ans = ans+(-1)**i*x**(2*i)/denom
    return ans
def se3_to_SE3(w,v): 
    # Создание нулевой матрицы
    deltaT = torch.zeros((4,4)).cuda()
    # Вычисление кососимметричной матрицы
    wx = skew_symmetric(w)
    # Вычисление углов вращения и вспомогательных матриц
    theta = w.norm(dim=-1)
    I = torch.eye(3,device=w.device,dtype=torch.float32)
    A = taylor_A(theta)
    B = taylor_B(theta)
    C = taylor_C(theta)
    # Обновление матрицы вращения
    deltaT[:3, :3] = I+A*wx+B*wx@wx
    # Обновление матрицы трансляции
    V = I+B*wx+C*wx@wx
    deltaT[:3, 3] = V@v
    # Установка единичного элемента в нижний правый угол
    deltaT[3, 3] = 1.
    return deltaT
# def compute_se3_to_SE3(w, v):
#     """
#     Вычисляет гомогенную матрицу преобразования SE(3) из векторов вращения и трансляции.
    
#     Parameters:
#     w (torch.tensor): Вектор вращения (3,)
#     v (torch.tensor): Вектор трансляции (3,)
    
#     Returns:
#     torch.tensor: Гомогенная матрица преобразования SE(3) (4, 4)
#     """
#     """
#     Кососимметричная матрица представляет собой матрицу, которая содержит кросс-произведение вектора w
#     """
#     def skew_symmetric(w):
#         w0, w1, w2 = w.unbind(dim=-1)
#         O = torch.zeros_like(w0)
#         return torch.stack([
#             torch.stack([O, -w2, w1], dim=-1),
#             torch.stack([w2, O, -w0], dim=-1),
#             torch.stack([-w1, w0, O], dim=-1)
#         ], dim=-2)
#     """
#     Функция вычисляет значение ряда Тейлора для заданного значения x и коэффициентов ряда coeffs
#     """
#     def taylor_series(x, coeffs):
#         result = torch.zeros_like(x)
#         denom = 1.
#         for i, coeff in enumerate(coeffs):
#             if i > 0:
#                 denom *= coeff
#             result += (-1)**i * x**(2*i) / denom
#         return result

    
    
#     # Вычисление кососимметричной матрицы
#     wx = skew_symmetric(w)
#     # Вычисление углов вращения и вспомогательных матриц
#     A_coeffs = [1, 6, 120, 5040, 362880] # Coefficients for sin(x)/x series
#     B_coeffs = [2, 24, 720, 40320, 3628800] # Coefficients for (1-cos(x))/x**2 series
#     C_coeffs = [6, 120, 5040, 362880, 39916800] # Coefficients for (x-sin(x))/x**3 series
#     theta = w.norm(dim=-1)
#     """
#     Вычисление коэффициентов рядов Тейлора для sin(x)/x, (1-cos(x))/x^2 и (x-sin(x))/x^3: Коэффициенты рядов Тейлора используются для приближенного вычисления функций sin(x)/x, (1-cos(x))/x^2 и (x-sin(x))/x^3
#     """
#     A = taylor_series(theta, A_coeffs)
#     B = taylor_series(theta, B_coeffs)
#     C = taylor_series(theta, C_coeffs)
#     I = torch.eye(3, device=w.device, dtype=torch.float32)
#     # Создание нулевой матрицы
#     deltaT = torch.zeros((4, 4), device=w.device, dtype=torch.float32)
#     # Обновление матрицы вращения
#     deltaT[:3, :3] = I + A * wx + B * wx @ wx
#     # Обновление матрицы трансляции
#     V = I + B * wx + C * wx @ wx
#     deltaT[:3, 3] = V @ v
#     # Установка единичного элемента в нижний правый угол
#     deltaT[3, 3] = 1.
#     return deltaT

