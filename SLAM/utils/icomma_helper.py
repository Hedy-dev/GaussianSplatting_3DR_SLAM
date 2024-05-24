import torch
from LoFTR.src.loftr import LoFTR, default_cfg
from copy import deepcopy
import numpy as np
from typing import NamedTuple

# Load the pre-trained LoFTR model. For more details, please refer to https://github.com/zju3dv/LoFTR.
def load_LoFTR(ckpt_path:str,temp_bug_fix:bool):
    _default_cfg = deepcopy(default_cfg)
    _default_cfg['coarse']['temp_bug_fix'] = temp_bug_fix  # set to False when using the old ckpt
   
    LoFTR_model = LoFTR(config=_default_cfg)
    LoFTR_model.load_state_dict(torch.load(ckpt_path)['state_dict'])
    LoFTR_model= LoFTR_model.eval().cuda()
    
    return LoFTR_model
"""
функция создает матрицу гомогенного преобразования для вращения вокруг 
оси z на угол ψ, delta[2]/180.*np.pi
"""
rot_psi = lambda phi: np.array([
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1]])
"""
функция создает матрицу гомогенного преобразования для вращения вокруг 
оси y на угол θ, delta[1]/180.*np.pi
"""
rot_theta = lambda th: np.array([
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1]])


# TODO: заменить на обычные, а то капец
"""
функция создает матрицу гомогенного преобразования для вращения вокруг оси 
x на угол ϕ, delta[0]/180.*np.pi
"""
rot_phi = lambda psi: np.array([
        [np.cos(psi), -np.sin(psi), 0, 0],
        [np.sin(psi), np.cos(psi), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
"""
Функция создаёт гомогенную матрицу трансляции (смещение) в 3D-пространстве по осям
x, y и z,
где x, y, z = delta[3],delta[4],delta[5], а delta default="[30,10,10,0.1,0.1,0.1]"
"""
def trans_t_xyz(tx, ty, tz):
    T = np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])
    return T
"""
R_c2w(3x3) — матрица вращения, которая преобразует координаты из системы координат камеры в мировую систему координат.
T_w2c (3x1) — вектор трансляции, который преобразует координаты из мировой системы координат в систему координат камеры.
Функция создает 4x4 гомогенную матрицу RT_w2c, которая сочетает в себе вращение и трансляцию, затем вычисляет её обратную матрицу RT_c2w, которая выполняет обратное преобразование.
"""
def combine_3dgs_rotation_translation(R_c2w, T_w2c):
    # Создается 4x4 единичная матрица RT_w2c
    RT_w2c = np.eye(4)
    # Верхний левый 3x3 блок матрицы RT_w2c заполняется транспонированной матрицей вращения R_c2w.T
    RT_w2c[:3, :3] = R_c2w.T
    # Верхний правый 3x1 блок матрицы RT_w2c заполняется вектором трансляции T_w2c
    RT_w2c[:3, 3] = T_w2c
    # Вычисляется обратная матрица RT_w2c, которая является матрицей преобразования из мировой системы координат в систему координат камеры RT_c2w
    RT_c2w=np.linalg.inv(RT_w2c)
    return RT_c2w

def create_transformation_matrix(delta):
    tx, ty, tz = delta[3], delta[4], delta[5]
    psi, theta, phi = delta[0] / 180. * np.pi, delta[1] / 180. * np.pi, delta[2] / 180. * np.pi

    T = np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])

    R_psi = np.array([
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1]
    ])

    R_theta = np.array([
        [np.cos(theta), 0, -np.sin(theta), 0],
        [0, 1, 0, 0],
        [np.sin(theta), 0, np.cos(theta), 0],
        [0, 0, 0, 1]
    ])

    R_phi = np.array([
        [np.cos(psi), -np.sin(psi), 0, 0],
        [np.sin(psi), np.cos(psi), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    transformation_matrix = T @ R_phi @ R_theta @ R_psi
    return transformation_matrix

def get_pose_estimation_input(obs_view,delta):
    # Исходная поза камеры
    gt_pose_c2w=combine_3dgs_rotation_translation(obs_view.R,obs_view.T)
    # Новая поза камеры в мировой системе координат после применения заданных вращений и трансляции
    # start_pose_c2w =  trans_t_xyz(delta[3],delta[4],delta[5]) @ rot_phi(delta[0]/180.*np.pi) @ rot_theta(delta[1]/180.*np.pi) @ rot_psi(delta[2]/180.*np.pi)  @ gt_pose_c2w
    start_pose_c2w = create_transformation_matrix(delta) @ gt_pose_c2w
    icomma_info = iComMa_input_info(gt_pose_c2w=gt_pose_c2w,
        start_pose_w2c=torch.from_numpy(np.linalg.inv(start_pose_c2w)).float(),
        query_image= obs_view.original_image[0:3, :, :],
        FoVx=obs_view.FoVx,
        FoVy=obs_view.FoVy,
        image_width=obs_view.image_width,
        image_height=obs_view.image_height)
    
    return icomma_info

class iComMa_input_info(NamedTuple):
    start_pose_w2c:torch.tensor
    gt_pose_c2w:np.array
    query_image:torch.tensor
    FoVx:float
    FoVy:float
    image_width:int
    image_height:int

    