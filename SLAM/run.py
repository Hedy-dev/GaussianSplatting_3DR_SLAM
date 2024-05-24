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

                
def camera_pose_estimation(gaussians:GaussianModel, background:torch.tensor, pipeline:PipelineParams, icommaparams:iComMaParams, icomma_info, output_path):
    # start pose & gt pose
    gt_pose_c2w=icomma_info.gt_pose_c2w
    start_pose_w2c=icomma_info.start_pose_w2c.cuda()
    
    # query_image for comparing 
    query_image = icomma_info.query_image.cuda()

    # initialize camera pose object
    camera_pose = Camera_Pose(start_pose_w2c,FoVx=icomma_info.FoVx,FoVy=icomma_info.FoVy,
                            image_width=icomma_info.image_width,image_height=icomma_info.image_height)
    camera_pose.cuda()

    # store gif elements
    imgs=[]
    
    matching_flag= not icommaparams.deprecate_matching
    num_iter_matching = 0

    # start optimizing
    optimizer = optim.Adam(camera_pose.parameters(),lr = icommaparams.camera_pose_lr)
    
    for k in range(icommaparams.pose_estimation_iter):

        rendering = render(camera_pose,
                           gaussians, 
                           pipeline, 
                           background,
                           compute_grad_cov2d = icommaparams.compute_grad_cov2d)#["render"]

        if matching_flag:
            loss_matching = loss_loftr(query_image,
                                       rendering,
                                       LoFTR_model,
                                       icommaparams.confidence_threshold_LoFTR,
                                       icommaparams.min_matching_points)
            
            loss_comparing = loss_mse(rendering,query_image)
            
            if loss_matching is None:
                loss = loss_comparing
            else:  
                loss = icommaparams.lambda_LoFTR *loss_matching + (1-icommaparams.lambda_LoFTR)*loss_comparing
                if loss_matching<0.001:
                    matching_flag=False
                    
            num_iter_matching += 1
        else:
            loss_comparing = loss_mse(rendering,query_image)
            loss = loss_comparing
            
            new_lrate = icommaparams.camera_pose_lr * (0.6 ** ((k - num_iter_matching + 1) / 50))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
        
        # output intermediate results
        if (k + 1) % 20 == 0 or k == 0:
            print_stat(k, matching_flag, loss_matching, loss_comparing, 
                       camera_pose, gt_pose_c2w)
            # output images
            if icommaparams.OVERLAY is True:
                with torch.no_grad():
                    rgb = rendering.clone().permute(1, 2, 0).cpu().detach().numpy()
                    rgb8 = to8b(rgb)
                    ref = to8b(query_image.permute(1, 2, 0).cpu().detach().numpy())
                    filename = os.path.join(output_path, str(k)+'.png')
                    dst = cv2.addWeighted(rgb8, 0.7, ref, 0.3, 0)
                    imageio.imwrite(filename, dst)
                    imgs.append(dst)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        camera_pose(start_pose_w2c)

    # output gif
    if icommaparams.OVERLAY is True:
        imageio.mimwrite(os.path.join(output_path, 'video.gif'), imgs, fps=4)
  
if __name__ == "__main__":

    args, model, pipeline, icommaparams = get_combined_args()

    makedirs(args.output_path, exist_ok=True)
    
    # load LoFTR_model
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


    LoFTR_model=load_LoFTR(icommaparams.LoFTR_ckpt_path,icommaparams.LoFTR_temp_bug_fix)
    
    # load gaussians
    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # get camera info from Scene
    # Reused 3DGS code to obtain camera information. 
    # You can customize the iComMa_input_info in practical applications.
    scene = Scene(dataset,gaussians,load_iteration=args.iteration,shuffle=False)
    obs_view=scene.getTestCameras()[args.obs_img_index]
    #obs_view=scene.getTrainCameras()[args.obs_img_index]
    icomma_info=get_pose_estimation_input(obs_view,ast.literal_eval(args.delta))
    
    # pose estimation
    camera_pose_estimation(gaussians,background,pipeline,icommaparams,icomma_info,args.output_path)