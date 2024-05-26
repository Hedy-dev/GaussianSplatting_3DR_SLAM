import torch
import numpy as np
import random
from scene import Scene
import torch.optim as optim
from os import makedirs
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams,iComMaParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.icomma_helper import load_LoFTR, get_pose_estimation_input
from utils.general_utils import print_stat
from utils.image_utils import to8b
import cv2
import imageio
import os
import ast
from scene.cameras import Camera_Pose
from utils.loss_utils import loss_loftr,loss_mse
from utils.system_utils import mkdir_p

def camera_pose_estimation(gaussians:GaussianModel, background:torch.tensor, pipeline:PipelineParams, icommaparams:iComMaParams, icomma_info, output_path):
    # start pose & gt pose
    # Гомогенная матрица преобразования из системы координат камеры в мировую систему координат
    gt_pose_c2w=icomma_info.gt_pose_c2w
    # Перемещение тензора start_pose_w2c, содержащего информацию о начальной позе камеры относительно мира, на устройство CUDA (if true)
    start_pose_w2c=icomma_info.start_pose_w2c.cuda()
    # Перемещение тензора query_image, содержащего информацию об изображении запроса
    query_image = icomma_info.query_image.cuda()

    
    camera_pose = Camera_Pose(start_pose_w2c,FoVx=icomma_info.FoVx,FoVy=icomma_info.FoVy,
                            image_width=icomma_info.image_width,image_height=icomma_info.image_height)
    camera_pose.cuda()

    # store gif elements
    imgs=[]
    ply_files=[]

    matching_flag= not icommaparams.deprecate_matching # Стоит ли объявлять соответствующий модуль устаревшим с самого начала True
    num_iter_matching = 0

    # start optimizing
    #optimizer = optim.Adam(camera_pose.parameters(),lr = icommaparams.camera_pose_lr)
    optimizer = optim.AdamW(camera_pose.parameters(), lr=icommaparams.camera_pose_lr)
    for k in range(icommaparams.pose_estimation_iter):
        # Выполняется рендеринг сцены с текущей камерой
        rendering = render(camera_pose,
                           gaussians, 
                           pipeline, 
                           background,
                           compute_grad_cov2d = icommaparams.compute_grad_cov2d)#["render"]
        # Если флаг matching_flag равен True (то есть модуль сопоставления не устарел)
        if matching_flag:
            # вычисление функции потерь с помощью модели LoFTR
            loss_matching = loss_loftr(query_image,
                                       rendering,
                                       LoFTR_model,
                                       icommaparams.confidence_threshold_LoFTR,
                                       icommaparams.min_matching_points)
            # функция потерь сравнения вычисляет среднеквадратичное отклонение между текущим рендерингом и изображением
            loss_comparing = loss_mse(rendering,query_image)
            
            if loss_matching is None:
                loss = loss_comparing
            else:  
                # Если loss_matching меньше некоторого порога (0.001), то флаг matching_flag устанавливается в False, что означает, что модуль сопоставления становится устаревшим
                loss = icommaparams.lambda_LoFTR *loss_matching + (1-icommaparams.lambda_LoFTR)*loss_comparing
                if loss_matching<0.001:
                    matching_flag=False
                    
            num_iter_matching += 1
        # если модуль сопоставления устарел, используется только функция потерь сравнения
        else:
            loss_comparing = loss_mse(rendering,query_image)
            loss = loss_comparing
            # новый learning rate для оптимизатора optimizer, который уменьшается со временем
            new_lrate = icommaparams.camera_pose_lr * (0.6 ** ((k - num_iter_matching + 1) / 50))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
        
        # вывод промежуточных результатов
        if (k + 1) % 20 == 0 or k == 0:
            print_stat(k, matching_flag, loss_matching, loss_comparing, 
                       camera_pose, gt_pose_c2w)
            # выводятся промежуточные результаты, включая значения функций потерь и визуализации
            if icommaparams.OVERLAY is True:
                with torch.no_grad():
                    rgb = rendering.clone().permute(1, 2, 0).cpu().detach().numpy()
                    rgb8 = to8b(rgb)
                    ref = to8b(query_image.permute(1, 2, 0).cpu().detach().numpy())
                    filename = os.path.join(output_path, str(k)+'.png')
                    filename2 = os.path.join(output_path, str(k)+'.ply')
                    dst = cv2.addWeighted(rgb8, 1.0, ref, 0.0, 0)
                    imageio.imwrite(filename, dst)
                    imgs.append(dst)
                    #ply_files.append(ply_file) # PlyData([el])
                    #gaussians.save_ply(filename, )
                  
                    #mkdir_p(os.path.dirname(filename))

                    #xyz = gaussians._xyz.detach().cpu().numpy()
                    #normals = np.zeros_like(xyz)
                    #f_dc = gaussians._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
                    #f_rest = gaussians._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
                    #opacities = gaussians._opacity.detach().cpu().numpy()
                    #scale = gaussians._scaling.detach().cpu().numpy()
                    #rotation = gaussians._rotation.detach().cpu().numpy()
#
                    #dtype_full = [(attribute, 'f4') for attribute in gaussians.construct_list_of_attributes()]
#
                    #elements = np.empty(xyz.shape[0], dtype=dtype_full)
                    #attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
                    #elements[:] = list(map(tuple, attributes))
                    #el = PlyElement.describe(elements, 'vertex')

                    #ply_file.write(filename2)
        """
        обнуляются градиенты, вычисляются градиенты функции потерь, и производится шаг оптимизации с помощью оптимизатора
        """
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # обновляется камера с использованием обновленных параметров
        camera_pose(start_pose_w2c)

    # output gif
    if icommaparams.OVERLAY is True:
        imageio.mimwrite(os.path.join(output_path, 'video.gif'), imgs, fps=4)
  
if __name__ == "__main__":
    # Возвращает словарь со всеми перечисленными аргументами, cfg_args
    args, model, pipeline, icommaparams = get_combined_args()
    # Создание директории по output_path
    makedirs(args.output_path, exist_ok=True)
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

    # Загрузка модели лофтера
    LoFTR_model=load_LoFTR(icommaparams.LoFTR_ckpt_path,icommaparams.LoFTR_temp_bug_fix)
    
    # Загрузка гауссиан
    # Вытаскиваем всю кучу параметров
    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # get camera info from Scene
    # Reused 3DGS code to obtain camera information. 
    # You can customize the iComMa_input_info in practical applications.
    scene = Scene(dataset,gaussians,load_iteration=args.iteration,shuffle=False)
    # Объект камеры, который соответствует индексу фотографии, указанному при запуске
    obs_view=scene.getTestCameras()[args.obs_img_index]
    #obs_view=scene.getTrainCameras()[args.obs_img_index]
    icomma_info=get_pose_estimation_input(obs_view,ast.literal_eval(args.delta))
    
    # pose estimation
    camera_pose_estimation(gaussians,background,pipeline,icommaparams,icomma_info,args.output_path)