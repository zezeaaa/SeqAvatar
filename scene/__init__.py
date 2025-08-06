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

import os
import torch
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.system_utils import mkdir_p
from utils.smpl_utils import dict_to_device
from utils.general_utils import generate_time_steps

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        # generate multiple time steps
        time_steps = generate_time_steps(args.minimal_time_step, args.max_time_step, 1, args.time_step_num, args.seq_len)

        if 'ZJU-MoCap' in args.source_path: 
            print("Assuming ZJU-MoCap dataset!")
            scene_info = sceneLoadTypeCallbacks["ZJU_MoCap"](args.source_path, args.white_background, args.eval, time_steps)
        elif 'I3D-Human' in args.source_path: 
            print("Assuming I3D-Human dataset!")
            scene_info = sceneLoadTypeCallbacks["I3DHuman"](args.source_path, args.white_background, args.eval, time_steps)
        elif 'DNA-Rendering' in args.source_path:
            print("Assuming DNA-Rendering dataset!")
            scene_info = sceneLoadTypeCallbacks["DNARendering"](args.source_path, args.white_background, args.eval, time_steps)
        else:
            assert False, "Could not recognize scene type!"
        
        # move to the device
        self.gaussians.canon_params = dict_to_device(scene_info.canon_params, args.data_device)
        self.gaussians.canon_vertices = torch.tensor(scene_info.canon_vertices).to(args.data_device).unsqueeze(0)
        self.gaussians.smpl_params_dict = dict_to_device(scene_info.smpl_params_dict, args.data_device)
        self.gaussians.cond_dict = dict_to_device(scene_info.cond_dict, args.data_device)

        # get transformation from A pose to T pose
        self.gaussians.get_canon2Tpose_transform(self.gaussians.canon_params)


        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for key in scene_info.test_cameras.keys():
                if scene_info.test_cameras[key]:
                    camlist.extend(scene_info.test_cameras[key])
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            for key in scene_info.test_cameras.keys():
                random.shuffle(scene_info.test_cameras[key])
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            
            self.test_cameras.setdefault(resolution_scale, {})
            for key in scene_info.test_cameras.keys():
                print(f'Loading Test {key} Cameras')
                self.test_cameras[resolution_scale][key] = cameraList_from_camInfos(scene_info.test_cameras[key], resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path, "point_cloud",
                                                    "iteration_" + str(self.loaded_iter),
                                                    "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

        if self.gaussians.motion_offset_flag:
            model_path = os.path.join(self.model_path, "mlp_ckpt", "iteration_" + str(self.loaded_iter), "ckpt.pth")
            if os.path.exists(model_path):
                ckpt = torch.load(model_path, map_location='cuda')
                self.gaussians.pose_decoder.load_state_dict(ckpt['pose_decoder'])
                self.gaussians.lweight_offset_decoder.load_state_dict(ckpt['lweight_offset_decoder'])
                
                if self.gaussians.non_rigid_flag:
                    self.gaussians.non_rigid_deformer.load_state_dict(ckpt['non_rigid_deformer'])

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

        if self.gaussians.motion_offset_flag:
            model_path = os.path.join(self.model_path, "mlp_ckpt", "iteration_" + str(iteration), "ckpt.pth")
            mkdir_p(os.path.dirname(model_path))

            save_dict = {
                'iter': iteration,
                'pose_decoder': self.gaussians.pose_decoder.state_dict(),
                'lweight_offset_decoder': self.gaussians.lweight_offset_decoder.state_dict(),
            }

            if self.gaussians.non_rigid_flag:
                save_dict['non_rigid_deformer'] = self.gaussians.non_rigid_deformer.state_dict()
            
            torch.save(save_dict, model_path)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
