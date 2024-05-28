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


rot_psi = lambda phi: np.array([
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1]])

rot_theta = lambda th: np.array([
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1]])


# TODO: заменить на обычные, а то капец

rot_phi = lambda psi: np.array([
        [np.cos(psi), -np.sin(psi), 0, 0],
        [np.sin(psi), np.cos(psi), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

def trans_t_xyz(tx, ty, tz):
    T = np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])
    return T

def draw_camera_in_top_camera(icomma_info, viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, compute_grad_cov2d=True):
    
    # Пример матрицы start_pose_c2w
    start_pose_c2w = torch.tensor(viewpoint_camera, dtype=torch.float32).cuda()
    
    # Преобразование от мира к камере B (обратная матрица к start_pose_c2w)
    # Предположим, что это просто нихуя не тождественное преобразование
    world_to_cameraB = torch.tensor(np.linalg.inv(
        #np.eye(4)
        # rot_phi - поворот вокруг оптической оси камеры
        # rot_theta - поворот "налево"
        # rot_psi - поворот вверх-вниз
        trans_t_xyz(0,-10,0) @ rot_phi(0/180.*np.pi) @ rot_theta(0/180.*np.pi) @ rot_psi(-90/180.*np.pi)
        ), dtype=torch.float32).cuda()
    camera_pose = Camera_Pose(world_to_cameraB,FoVx=icomma_info.FoVx,FoVy=icomma_info.FoVy,
                            image_width=icomma_info.image_width,image_height=icomma_info.image_height)
    camera_pose.cuda()
    # camera_pose для камеры с видом сверху
    camera_b_view = render(camera_pose,
                           gaussians, 
                           pipeline, 
                           background,
                           compute_grad_cov2d = icommaparams.compute_grad_cov2d)

    # Положение камеры в пространстве B
    # print(type(world_to_cameraB), type(start_pose_c2w))
    cameraB_pose = world_to_cameraB @ start_pose_c2w

    # Параметры проекции камеры B
    # Фокусное расстояние, координаты центра изображения, коэффициенты искажения и т. д.
    # Предположим, что они известны
    focal_length = 200 

    image_center = torch.tensor(np.array([camera_b_view.shape[1] / 2, camera_b_view.shape[2] / 2]), dtype=torch.float32).cuda()  # Пример координат центра изображения
    distortion_coeffs = np.zeros(5)  # Пример коэффициентов искажения

    # Преобразование координат камеры в пространстве B в координаты на изображении
    # Это может быть выполнено с использованием функции проекции, например, функции cv2.projectPoints в OpenCV
    # Здесь просто приведен пример для наглядности
    camera_coordinates_B = cameraB_pose[:3, 3]
    image_coordinates_B = (focal_length * camera_coordinates_B[:2] / camera_coordinates_B[2]) + image_center

    # print("Координаты камеры на изображении с камеры B:", image_coordinates_B)
    # print("camera_b_view = ", camera_b_view.shape)
    # rgb = camera_b_view.clone().permute(1, 2, 0).cpu().detach().numpy()
    # rgb8 = to8b(rgb)
    # filename = os.path.join('rendering.png')
    # imageio.imwrite(filename, rgb8)
    image_coordinates_B_cort = image_coordinates_B.clone().cpu().detach().numpy()
    image_coordinates_B_cort = (int(image_coordinates_B_cort[0]), int(image_coordinates_B_cort[1]))
    # return_image = to8b(camera_b_view.clone().permute(2, 1, 0).cpu().detach().numpy().astype(np.uint8))
    # return_image = return_image.copy()
    # print('test_image_smth shape = ', test_image_smth.shape, type(test_image_smth))
    # return_image = cv2.circle(return_image, cam_centre, 5, (0,255,0), thickness=1, lineType=8, shift=0)
    return  image_coordinates_B_cort
    #return_image,


def camera_pose_estimation(gaussians:GaussianModel, background:torch.tensor, pipeline:PipelineParams, icommaparams:iComMaParams, icomma_info, output_path):
    # start pose & gt pose
    # Гомогенная матрица преобразования из системы координат камеры в мировую систему координат
    gt_pose_c2w=icomma_info.gt_pose_c2w
    # Перемещение тензора start_pose_w2c, содержащего информацию о начальной позе камеры относительно мира, на устройство CUDA (if true)
    start_pose_w2c=icomma_info.start_pose_w2c.cuda()
    # Перемещение тензора query_image, содержащего информацию об изображении запроса
    # query_image = 
    query_image = icomma_info.query_image.cuda()

    
    camera_pose = Camera_Pose(start_pose_w2c,FoVx=icomma_info.FoVx,FoVy=icomma_info.FoVy,
                            image_width=icomma_info.image_width,image_height=icomma_info.image_height)
    #print("start_pose_w2c: ", start_pose_w2c)
    camera_pose.cuda()
    ## Настройки
    camera_poses_sequence = []


    # store gif elements
    imgs=[]
    ply_files=[]

    matching_flag= not icommaparams.deprecate_matching # Стоит ли объявлять соответствующий модуль устаревшим с самого начала True
    num_iter_matching = 0

    # start optimizing
    optimizer = optim.Adam(camera_pose.parameters(),lr = icommaparams.camera_pose_lr)
    #optimizer = optim.RAdam(camera_pose.parameters(), lr=icommaparams.camera_pose_lr)
    start_pose_w2c_r = torch.tensor([[5.9492171e-01,  3.0355409e-01, -7.4426007e-01,  5.2494292e+00], [2.2395043e-02,  9.1932631e-01,  3.9285806e-01, -1.2358599e+00], [8.0347174e-01, -2.5038749e-01,  5.4012907e-01, -1.6433660e+00], [-4.1564192e-09, -2.1902808e-08, -2.1574653e-08,  9.9999988e-01]], device='cuda:0').cuda()
    camera_pose_r = Camera_Pose(start_pose_w2c_r,FoVx=icomma_info.FoVx,FoVy=icomma_info.FoVy,
                            image_width=icomma_info.image_width,image_height=icomma_info.image_height)
    camera_pose_r.cuda()
    for k in range(icommaparams.pose_estimation_iter):
        # Выполняется рендеринг сцены с текущей камерой
        rendering = render(camera_pose,
                           gaussians, 
                           pipeline, 
                           background,
                           compute_grad_cov2d = icommaparams.compute_grad_cov2d)#["render"]
        rendering2 = render(camera_pose_r,
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
                        # output images
            matrix_pose_c2w_to_top_camera = camera_pose.current_campose_c2w()
            current_camera_pose = draw_camera_in_top_camera(icomma_info, matrix_pose_c2w_to_top_camera, gaussians, 
                           pipeline, 
                           background,
                           compute_grad_cov2d = icommaparams.compute_grad_cov2d)
            # a
            camera_poses_sequence.append(current_camera_pose)
            # camera_b_view_query.append(a)
            # print(a)
            print("current_campose_c2w = ", matrix_pose_c2w_to_top_camera)

            # выводятся промежуточные результаты, включая значения функций потерь и визуализации
            if icommaparams.OVERLAY is True:
                with torch.no_grad():
                    rgb = rendering.clone().permute(1, 2, 0).cpu().detach().numpy()
                    rgb8 = to8b(rgb)

                    ref = to8b(query_image.permute(1, 2, 0).cpu().detach().numpy())
                    filename = os.path.join(output_path, str(k)+'.png')
                    #filename2 = os.path.join(output_path, str(k)+'.ply')
                    dst = cv2.addWeighted(rgb8, 1.0, ref, 0.0, 0)
                    imageio.imwrite(filename, dst)
                    imgs.append(dst)
                    #ply_files.append(ply_file) # PlyData([el])
                    #gaussians.save_ply(filename, )
                    rgb_r = rendering2.clone().permute(1, 2, 0).cpu().detach().numpy()
                    rgb8_r = to8b(rgb_r)

                    ref = to8b(query_image.permute(1, 2, 0).cpu().detach().numpy())
                    filename_r = os.path.join(output_path, str(k)+'ref.png')
                    #filename2 = os.path.join(output_path, str(k)+'.ply')
                    dstr = cv2.addWeighted(rgb8_r, 1.0, ref, 0.0, 0)
                    #imageio.imwrite(filename_r, dstr)
                    #imgs.append(dstr)
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
    world_to_cameraB = torch.tensor(np.linalg.inv(
            #np.eye(4)
            # rot_phi - поворот вокруг оптической оси камеры
            # rot_theta - поворот "налево"
            # rot_psi - поворот вверх-вниз
            trans_t_xyz(0,-10,0) @ rot_phi(0/180.*np.pi) @ rot_theta(0/180.*np.pi) @ rot_psi(-90/180.*np.pi)
            ), dtype=torch.float32).cuda()
    camera_pose = Camera_Pose(world_to_cameraB,FoVx=icomma_info.FoVx,FoVy=icomma_info.FoVy,
                                image_width=icomma_info.image_width,image_height=icomma_info.image_height)
    camera_b_view = to8b(render(camera_pose,
                               gaussians, 
                               pipeline, 
                               background,
                               compute_grad_cov2d = icommaparams.compute_grad_cov2d).clone().permute(1, 2, 0).cpu().detach().numpy())
        # return_image = to8b(camera_b_view.clone().permute(2, 1, 0).cpu().detach().numpy().astype(np.uint8))
    camera_b_view = camera_b_view.copy()
    for camera_poses_point in camera_poses_sequence:
        cv2.circle(camera_b_view, camera_poses_point, 5, (0,255,0), thickness=1, lineType=8, shift=0)
    cv2.imwrite('camera_path.png', camera_b_view)
    # output gif
    if icommaparams.OVERLAY is True:
        imageio.mimwrite(os.path.join(output_path, 'video.gif'), imgs, fps=4)
        ref = to8b(query_image.permute(1, 2, 0).cpu().detach().numpy())
        filename = os.path.join('ref.png')
        imageio.imwrite(filename, ref)
  
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


    scene = Scene(dataset,gaussians,load_iteration=args.iteration,shuffle=False)
    # Объект камеры, который соответствует индексу фотографии, указанному при запуске
    obs_view=scene.getTestCameras()[args.obs_img_index]
    #obs_view=scene.getTrainCameras()[args.obs_img_index]
    icomma_info=get_pose_estimation_input(obs_view,ast.literal_eval(args.delta))
    
    # pose estimation
    camera_pose_estimation(gaussians,background,pipeline,icommaparams,icomma_info,args.output_path)