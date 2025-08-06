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
from PIL import Image
from typing import NamedTuple
from utils.graphics_utils import focal2fov, getNerfppNorm, BasicPointCloud
import numpy as np
import torch
import cv2
from utils.sh_utils import SH2RGB
from smpl_model.smpl.smpl_numpy import SMPL
from smpl_model.smplx.body_models import SMPLX
from utils.SMCReader import SMCReader
from utils.transformation_util import axis_angle_to_matrix, matrix_to_axis_angle
from utils.general_utils import storePly, fetchPly
from utils.smpl_utils import batch_rodrigues, create_canonical_vertices
from utils.image_utils import load_image_mask
import pickle

class CameraInfo(NamedTuple):
    uid: int
    pose_id: int
    R: np.array
    T: np.array
    K: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    bkgd_mask: np.array
    bound_mask: np.array
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    canon_params: dict
    canon_vertices: np.array
    smpl_params_dict: dict
    cond_dict: dict

##################################   ID3-Human   ##################################
ID3HUMAN_CFG = {
    'ID1_1': {'train_view': [1, 3, 6, 8], 'test_view': [2, 4, 7, 9], 'interval': 1},
    'ID1_2': {'train_view': [0, 2, 5, 7], 'test_view': [1, 6, 8, 9], 'interval': 1},
    'ID2_1': {'train_view': [1, 3, 5, 7, 9], 'test_view': [2, 4, 6, 10], 'interval': 3},
    'ID2_2': {'train_view': [1, 3, 5, 7, 9], 'test_view': [2, 4, 6, 8, 10], 'interval': 3},
    'ID3_1': {'train_view': [1, 3, 5, 7, 9], 'test_view': [2, 4, 6, 8, 10], 'interval': 3},
    'ID3_2': {'train_view': [1, 3, 5, 7, 9], 'test_view': [2, 4, 6, 8, 10], 'interval': 3},
}

def readID3HumanInfo(path, white_background, eval, time_steps):
    scene_name = os.path.basename(path).split('-')[0]
    if scene_name not in ID3HUMAN_CFG.keys():
        raise ValueError('Unknown dataset')
    
    # camera view splitting, follow Dyco's setting
    train_view, test_view, interval = ID3HUMAN_CFG[scene_name]['train_view'], ID3HUMAN_CFG[scene_name]['test_view'], ID3HUMAN_CFG[scene_name]['interval']
    
    # load SMPL model
    smpl_model = SMPL(sex='neutral', model_dir='smpl_model/models/')

    # build canonical space SMPL points
    canon_params, canon_vertices = create_canonical_vertices(smpl_model, 'smpl')

    # read cameras
    test_cam_infos = {}
    smpl_params_dict, cond_dict = {}, {} # observation space smpl params, conditions for non-rigid deformation
    delta_pose_xyz_cache = {} # cache for sequential condition calculation
    
    print("Reading Training Transforms")
    train_cam_infos = readCamerasI3DHuman(path, smpl_model, train_view, white_background, split='train', 
                        interval=interval, time_steps=time_steps, 
                        smpl_params_dict=smpl_params_dict, cond_dict=cond_dict,
                        delta_pose_xyz_cache=delta_pose_xyz_cache)

    print("Reading Test Novelview Transforms")
    test_cam_infos['novelview'] = readCamerasI3DHuman(path, smpl_model, test_view, white_background, split='novelview', 
                                    interval=interval, time_steps=time_steps, 
                                    smpl_params_dict=smpl_params_dict, cond_dict=cond_dict,
                                    delta_pose_xyz_cache=delta_pose_xyz_cache)
    
    print("Reading Test Novelpose Transforms")
    test_cam_infos['novelpose'] = readCamerasI3DHuman(path, smpl_model, test_view, white_background, split='novelpose', 
                                    interval=interval, time_steps=time_steps, 
                                    smpl_params_dict=smpl_params_dict, cond_dict=cond_dict,
                                    delta_pose_xyz_cache=delta_pose_xyz_cache)

    if not eval:
        for key in test_cam_infos.keys():
            train_cam_infos.extend(test_cam_infos[key])
            test_cam_infos[key] = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if len(train_view) == 1:
        nerf_normalization['radius'] = 1

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = canon_vertices.shape[0]
        print(f"Generating random point cloud ({num_pts})...")
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=canon_vertices, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, canon_vertices, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos, test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization, ply_path=ply_path,
                           canon_params=canon_params, canon_vertices=canon_vertices,
                           smpl_params_dict=smpl_params_dict, cond_dict=cond_dict)
    return scene_info

def readCamerasI3DHuman(path, smpl_model, output_view, white_background, image_scaling=1.0, split='train', interval=1, time_steps=None, smpl_params_dict=None, cond_dict=None, delta_pose_xyz_cache=None, multi=300.0):
    cam_infos = []
    scene_name = os.path.basename(path).split('-')[0]
    scene_dir = os.path.join(os.path.dirname(path), scene_name + '-' + split)
    with open(os.path.join(scene_dir, 'cameras.pkl'), 'rb') as f: 
        cameras = pickle.load(f)
    with open(os.path.join(scene_dir, 'mesh_infos.pkl'), 'rb') as f:   
        mesh_infos = pickle.load(f)
    with open(os.path.join(scene_dir, 'frameid_pose.pkl'), 'rb') as f:   
        frameid_pose = pickle.load(f)

    frame_ids = sorted(list(set([int(img_name.split('_')[1]) for img_name in cameras.keys()])))

    # define how to read each pose_id's pose and xyz in I3D-Human dataset
    def get_pose_xyz_func(pose_id): 
        data = frameid_pose[pose_id]
        rh, poses_data, th = data['Rh'], data['poses'], data['Th']
        pose = np.concatenate([rh[None, :], poses_data], axis=0)
        pose_mat = axis_angle_to_matrix(torch.from_numpy(pose))

        full_poses = np.zeros(72, dtype=np.float32)
        full_poses[3:] = poses_data.flatten()
        R_mat = cv2.Rodrigues(rh[None, :])[0].astype(np.float32)

        shapes = np.zeros((1,10), dtype=np.float32)
        xyz, _ = smpl_model(full_poses[None, :].astype(np.float32), shapes.reshape(-1))
        xyz_transformed = (xyz @ R_mat.T + th[None, :]).astype(np.float32)
        
        return pose_mat, xyz_transformed
    
    for idx, pose_index in enumerate(frame_ids):
        frame_name = 'frame_{:06d}_view_{:02d}'.format(pose_index, output_view[0])

        # load SMPL data in observation space
        smpl_id = pose_index // interval
        if smpl_id not in smpl_params_dict.keys():
            smpl_param = {}
            Rh = mesh_infos[frame_name]['Rh'][None, :]
            smpl_param['Rh'] = Rh
            smpl_param['R'] = cv2.Rodrigues(Rh)[0].astype(np.float32)
            smpl_param['Th'] = mesh_infos[frame_name]['Th'][None, :]
            smpl_param['poses'] = mesh_infos[frame_name]['poses'][None, :]
            smpl_param['joints'] = mesh_infos[frame_name]['joints']
            smpl_param['shapes'] = np.zeros((1,10), dtype=np.float32)
            smpl_param['rot_mats'] = batch_rodrigues(torch.from_numpy(smpl_param['poses']).view(-1, 3)).view([1, -1, 3, 3])
            smpl_params_dict[smpl_id] = smpl_param
            
        smpl_param = smpl_params_dict[smpl_id]

        # load conditions
        if smpl_id not in cond_dict.keys():
            pose_conds = torch.from_numpy(frameid_pose[smpl_id]['poses']).unsqueeze(0)
            seq_pose_conds, seq_xyz_conds = get_seq_pose_xyz_cond(pose_index, time_steps, interval, get_pose_xyz_func, delta_pose_xyz_cache, multi=multi)
            cond_dict[smpl_id] = {'pose_conds': pose_conds, 'seq_pose_conds': seq_pose_conds, 'seq_xyz_conds': seq_xyz_conds}

        conds = cond_dict[smpl_id]
        pose_conds, seq_pose_conds, seq_xyz_conds = conds['pose_conds'], conds['seq_pose_conds'], conds['seq_xyz_conds']

        for view_index in output_view:
            cam_id = idx * len(output_view) + view_index
            image_name = f'frame_{pose_index:06d}_view_{view_index:02d}'

            # Load image, mask, K, D, R, T
            image_path = os.path.join(scene_dir, 'images', image_name + '.png')
            bkgd_mask_path = os.path.join(scene_dir, 'masks', image_name + '.png')

            K = cameras[image_name]['intrinsics']
            R = cameras[image_name]['extrinsics'][:3, :3]
            T = cameras[image_name]['extrinsics'][:3, 3:4]

            image, bkgd_mask, bound_mask, K = load_image_mask(image_path, bkgd_mask_path, white_background, image_scaling, K)
            if bkgd_mask is None:
                continue

            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            w2c = np.eye(4)
            w2c[:3,:3] = R
            w2c[:3,3:4] = T

            # get the world-to-camera transform and set R, T
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            focalX, focalY, width, height = K[0, 0], K[1, 1], image.size[0], image.size[1]
            FovX, FovY = focal2fov(focalX, width), focal2fov(focalY, height)

            cam_infos.append(CameraInfo(uid=cam_id, pose_id=smpl_id, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, bkgd_mask=bkgd_mask, 
                            bound_mask=bound_mask, width=width, height=height))
    return cam_infos

##################################   DNA-Rendering   ##################################
def readDNARenderingInfo(path, white_background, eval, time_steps):
    scene_name = os.path.basename(path)
    main_path = os.path.join(path, scene_name + '.smc')
    smc_reader = SMCReader(main_path)

    # camera view splitting
    train_view = [i for i in range(0, 48, 2)]
    test_view = [i for i in range(48, 60, 2)]

    # load SMPL-X model
    gender = smc_reader.actor_info['gender']
    smpl_model = SMPLX('smpl_model/models/', smpl_type='smplx', gender=gender, 
                        use_face_contour=True, flat_hand_mean=False, use_pca=False,
                        num_pca_comps=24, num_betas=10, num_expression_coeffs=10, ext='pkl')
    
    # build canonical space SMPLX points
    canon_params, canon_vertices = create_canonical_vertices(smpl_model, 'smplx')

    test_cam_infos = {}
    smpl_params_dict, cond_dict = {}, {} # observation space smpl params, conditions for non-rigid deformation
    delta_pose_xyz_cache = {} # cache for sequential condition calculation

    # read cameras
    print("Reading Training Transforms")
    train_cam_infos = readCamerasDNARendering(path, train_view, white_background, split='train', time_steps=time_steps,
                        smpl_params_dict=smpl_params_dict, cond_dict=cond_dict, 
                        delta_pose_xyz_cache=delta_pose_xyz_cache)

    print("Reading Novel View Transforms")
    test_cam_infos['novelview'] = readCamerasDNARendering(path, test_view, white_background, 
                                    split='novelview', time_steps=time_steps, 
                                    smpl_params_dict=smpl_params_dict, cond_dict=cond_dict,
                                    delta_pose_xyz_cache=delta_pose_xyz_cache)
    
    if not eval:
        for key in test_cam_infos.keys():
            train_cam_infos.extend(test_cam_infos[key])
            test_cam_infos[key] = []
    
    nerf_normalization = getNerfppNorm(train_cam_infos)
    if len(train_view) == 1:
        nerf_normalization['radius'] = 1
    
    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = canon_vertices.shape[0]
        print(f"Generating random point cloud ({num_pts})...")
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=canon_vertices, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, canon_vertices, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos, test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization, ply_path=ply_path,
                           canon_params=canon_params, canon_vertices=canon_vertices,
                           smpl_params_dict=smpl_params_dict, cond_dict=cond_dict)
    return scene_info

def readCamerasDNARendering(path, output_view, white_background, split='train', interval=1, time_steps=None, smpl_params_dict=None, cond_dict=None, delta_pose_xyz_cache=None, multi=300.0):
    cam_infos = []
    if split == 'train':
        pose_start, pose_interval, pose_num = 0, 1, 100
    elif split == 'novelview':
        pose_start, pose_interval, pose_num = 0, 5, 20
    else:
        raise ValueError('Unknown split type for DNA-Rendering dataset')
    
    # define how to read each pose_id's pose and xyz in DNA-Rendering dataset    
    def get_pose_xyz_func(pose_id):
        model_file = os.path.join(path, 'model', f'{pose_id:06d}.npz')
        smpl_param = np.load(model_file, allow_pickle=True)
        poses = smpl_param['poses'].reshape(-1, 3)
        pose_mat = axis_angle_to_matrix(torch.from_numpy(poses))
        return pose_mat, smpl_param['obs_xyz']
    
    for idx, pose_index in enumerate(range(pose_start, pose_start + pose_num * pose_interval, pose_interval)):
        # load smpl data 
        if pose_index not in smpl_params_dict.keys():
            model_file = os.path.join(path, 'model', f'{pose_index:06d}.npz')
            loaded_data = np.load(model_file, allow_pickle=True)
            smpl_param = {key: loaded_data[key] for key in loaded_data.files}
            smpl_param['rot_mats'] = batch_rodrigues(torch.from_numpy(smpl_param['poses']).view(-1, 3)).view([1, -1, 3, 3])
            smpl_params_dict[pose_index] = smpl_param

        smpl_param = smpl_params_dict[pose_index]

        # load condition
        if pose_index not in cond_dict.keys():
            pose_conds = smpl_param['poses'].reshape(-1,3)[1:]
            if not isinstance(pose_conds, torch.Tensor):
                pose_conds = torch.from_numpy(pose_conds).unsqueeze(0)

            seq_pose_conds, seq_xyz_conds = get_seq_pose_xyz_cond(pose_index, time_steps, interval, get_pose_xyz_func, delta_pose_xyz_cache, multi=multi)
            cond_dict[pose_index] = {'pose_conds': pose_conds, 'seq_pose_conds': seq_pose_conds, 'seq_xyz_conds': seq_xyz_conds}

        conds = cond_dict[pose_index]
        pose_conds, seq_pose_conds, seq_xyz_conds = conds['pose_conds'], conds['seq_pose_conds'], conds['seq_xyz_conds']

        for view_index in output_view:
            cam_id = idx * len(output_view) + view_index
            image_name = f'frame_{pose_index:06d}_view_{view_index:02d}'

            # Load camera, image, mask
            cam_path = os.path.join(path, 'cameras', f'{view_index:02d}', f'{pose_index:06d}.npz')
            img_path = os.path.join(path, 'images', f'{view_index:02d}', f'{pose_index:06d}.png')
            bkgd_mask_path = os.path.join(path, 'bkgd_masks', f'{view_index:02d}', f'{pose_index:06d}.png')

            cam_params = np.load(cam_path, allow_pickle=True)
            # D = cam_params['D']
            K = cam_params['K']
            R, T = cam_params['RT'][:3, :3], cam_params['RT'][:3, 3]
            c2w = np.eye(4)
            c2w[:3, :3] = R
            c2w[:3, 3:4] = T.reshape(-1, 1)

            image, bkgd_mask, bound_mask, _ = load_image_mask(img_path, bkgd_mask_path, white_background)

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            focalX, focalY, width, height = K[0, 0], K[1, 1], image.size[0], image.size[1]
            FovX, FovY = focal2fov(focalX, width), focal2fov(focalY, height)

            cam_infos.append(CameraInfo(uid=cam_id, pose_id=pose_index, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image,
                                        image_path=img_path, image_name=image_name, bkgd_mask=bkgd_mask,
                                        bound_mask=bound_mask, width=width, height=height))
    
    return cam_infos

##################################   ZJU-Mocap   ##################################
ZJUMOCAP_CFG = {
    'CoreView_377': {'train': {'interval': 1, 'num': 570},
                     'test':  {'interval': 30, 'num': 19}},
    'CoreView_386': {'train': {'interval': 1, 'num': 540},
                     'test':  {'interval': 30, 'num': 18}},
    'CoreView_387': {'train': {'interval': 1, 'num': 540},
                     'test':  {'interval': 30, 'num': 18}},
    'CoreView_392': {'train': {'interval': 1, 'num': 556},
                     'test':  {'interval': 30, 'num': 19}},
    'CoreView_393': {'train': {'interval': 1, 'num': 658},
                     'test':  {'interval': 30, 'num': 22}},
    'CoreView_394': {'train': {'interval': 1, 'num': 475},
                     'test':  {'interval': 30, 'num': 16}}
    }
def readZJUMoCapInfo(path, white_background, eval, time_steps):
    # camera view splitting
    train_view = [0]
    test_view = [i for i in range(1, 23)]

    # load SMPL model
    smpl_model = SMPL(sex='neutral', model_dir='smpl_model/models/')

    # build canonical space SMPL points
    canon_params, canon_vertices = create_canonical_vertices(smpl_model, 'smpl')

    test_cam_infos = {}
    smpl_params_dict, cond_dict = {}, {} # observation space smpl params, conditions for non-rigid deformation
    delta_pose_xyz_cache = {} # cache for sequential condition calculation
    

    # read cameras
    print("Reading Training Transforms")
    train_cam_infos = readCamerasZJUMoCap(path, train_view, white_background, split='train', time_steps=time_steps, 
                           smpl_params_dict=smpl_params_dict, cond_dict=cond_dict,
                           delta_pose_xyz_cache=delta_pose_xyz_cache)
    
    print("Reading Test Transforms")
    test_cam_infos['test'] = readCamerasZJUMoCap(path, test_view, white_background, split='test', time_steps=time_steps, 
                                smpl_params_dict=smpl_params_dict, cond_dict=cond_dict,
                                delta_pose_xyz_cache=delta_pose_xyz_cache)
    
    if not eval:
        for key in test_cam_infos.keys():
            train_cam_infos.extend(test_cam_infos[key])
            test_cam_infos[key] = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if len(train_view) == 1:
        nerf_normalization['radius'] = 1

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = canon_vertices.shape[0]
        print(f"Generating random point cloud ({num_pts})...")
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=canon_vertices, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, canon_vertices, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos, test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization, ply_path=ply_path,
                           canon_params=canon_params, canon_vertices=canon_vertices,
                           smpl_params_dict=smpl_params_dict, cond_dict=cond_dict)
    return scene_info

def readCamerasZJUMoCap(path, output_view, white_background, image_scaling=0.5, split='train', interval=1, time_steps=None, smpl_params_dict=None, cond_dict=None, delta_pose_xyz_cache=None):
    cam_infos = []
    pose_start = 0
    scene_name = os.path.basename(path)
    pose_interval, pose_num = ZJUMOCAP_CFG[scene_name][split]['interval'], ZJUMOCAP_CFG[scene_name][split]['num']

    ann_file = os.path.join(path, 'annots.npy')
    annots = np.load(ann_file, allow_pickle=True).item()
    cams = annots['cams']
    ims = np.array([
        np.array(ims_data['ims'])[output_view]
        for ims_data in annots['ims'][pose_start:pose_start + pose_num * pose_interval]
    ])

    cam_inds = np.array([
        np.arange(len(ims_data['ims']))[output_view]
        for ims_data in annots['ims'][pose_start:pose_start + pose_num * pose_interval]
    ])

    # define how to read each pose_id's pose and xyz in ZJU-Mocap dataset
    def get_pose_xyz_func(pose_id):
        smpl_param_path = os.path.join(path, "new_params", '{}.npy'.format(pose_id))
        loaded_data = np.load(smpl_param_path, allow_pickle=True).item()
        pose = np.concatenate([loaded_data['Rh'], loaded_data['poses'][:, 3:].reshape(-1, 3)], axis=0)
        vertices_path = os.path.join(path, 'new_vertices', '{}.npy'.format(pose_id))
        xyz = np.load(vertices_path).astype(np.float32)
        pose_mat = axis_angle_to_matrix(torch.from_numpy(pose))
        return pose_mat, xyz
    
    for idx, pose_index in enumerate(range(pose_start, pose_start + pose_num * pose_interval, pose_interval)):
        # load smpl data 
        if pose_index not in smpl_params_dict.keys():
            smpl_param = {}
            smpl_param_path = os.path.join(path, "new_params", '{}.npy'.format(pose_index))
            loaded_smpl_data = np.load(smpl_param_path, allow_pickle=True).item()
            Rh = loaded_smpl_data['Rh']
            smpl_param['Rh'] = Rh
            smpl_param['R'] = cv2.Rodrigues(Rh)[0].astype(np.float32)
            smpl_param['Th'] = loaded_smpl_data['Th'].astype(np.float32)
            smpl_param['shapes'] = loaded_smpl_data['shapes'].astype(np.float32)
            smpl_param['poses'] = loaded_smpl_data['poses'].astype(np.float32)
            smpl_param['rot_mats'] = batch_rodrigues(torch.from_numpy(smpl_param['poses']).view(-1, 3)).view([1, -1, 3, 3])
            smpl_params_dict[pose_index] = smpl_param

        smpl_param = smpl_params_dict[pose_index]

        if pose_index not in cond_dict.keys():
            pose_conds = torch.from_numpy(smpl_param['poses'][:, 3:].reshape(-1, 3)).unsqueeze(0)
            seq_pose_conds, seq_xyz_conds = get_seq_pose_xyz_cond(pose_index, time_steps, interval, get_pose_xyz_func, delta_pose_xyz_cache)
            cond_dict[pose_index] = {'pose_conds': pose_conds, 'seq_pose_conds': seq_pose_conds, 'seq_xyz_conds': seq_xyz_conds}

        conds = cond_dict[pose_index]
        pose_conds, seq_pose_conds, seq_xyz_conds = conds['pose_conds'], conds['seq_pose_conds'], conds['seq_xyz_conds']

        for view_index in range(len(output_view)):
            cam_id = idx * len(output_view) + view_index

            # Load image, mask, K, D, R, T
            image_path = os.path.join(path, ims[pose_index][view_index].replace('\\', '/'))
            bkgd_mask_path = os.path.join(path, 'mask_cihp', ims[pose_index][view_index].replace('\\', '/')).replace('jpg', 'png')
            image_name = ims[pose_index][view_index].split('.')[0]

            cam_ind = cam_inds[pose_index][view_index]
            K = np.array(cams['K'][cam_ind])
            D = np.array(cams['D'][cam_ind])
            R = np.array(cams['R'][cam_ind])
            T = np.array(cams['T'][cam_ind]) / 1000.

            image, bkgd_mask, bound_mask, K = load_image_mask(image_path, bkgd_mask_path, white_background, image_scaling, K, D)

            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            w2c = np.eye(4)
            w2c[:3,:3] = R
            w2c[:3,3:4] = T
            # get the world-to-camera transform and set R, T
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            focalX, focalY, width, height = K[0, 0], K[1, 1], image.size[0], image.size[1]
            FovX, FovY = focal2fov(focalX, width), focal2fov(focalY, height)

            image_name = f'frame_{pose_index:06d}_view_{view_index:02d}'
            cam_infos.append(CameraInfo(uid=cam_id, pose_id=pose_index, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, bkgd_mask=bkgd_mask, 
                            bound_mask=bound_mask, width=width, height=height))
    
    return cam_infos

def get_seq_pose_xyz_cond(pose_index, time_steps, interval, get_pose_xyz_func, delta_pose_xyz_cache, multi=1.0):
    seq_pose_conds, seq_xyz_conds = {}, {}
    for time_step, seq_len in time_steps.items():
        seq_pose_cond, seq_xyz_cond = [], []
        for i in range(seq_len):
            cur_id = max((pose_index - i * time_step) // interval, 0)
            former_id = max((pose_index - (i + 1) * time_step) // interval, 0)
            delta_key = str(cur_id) + '-' + str(former_id)

            if delta_key not in delta_pose_xyz_cache.keys():
                cur_pose_mat, cur_obs_xyz = get_pose_xyz_func(cur_id)
                former_pose_mat, former_obs_xyz = get_pose_xyz_func(former_id)

                delta_pose_mat = torch.matmul(cur_pose_mat, torch.linalg.inv(former_pose_mat))
                posedelta = matrix_to_axis_angle(delta_pose_mat)
                xyz_delta = cur_obs_xyz - former_obs_xyz
                delta_pose_xyz_cache[delta_key] = {'pose_delta': posedelta, 'xyz_delta': xyz_delta}
            
            posedelta = delta_pose_xyz_cache[delta_key]['pose_delta']
            xyz_delta = delta_pose_xyz_cache[delta_key]['xyz_delta']
            seq_pose_cond.append(posedelta)
            seq_xyz_cond.append(xyz_delta)

        seq_pose_conds[time_step] = torch.stack(seq_pose_cond, axis=0)
        seq_xyz_conds[time_step] = torch.from_numpy(np.stack(seq_xyz_cond, axis=1))
    
    seq_pose_conds = torch.stack([seq_pose_conds[time_step] for time_step in time_steps], axis=1).unsqueeze(0)
    seq_xyz_conds = (torch.stack([seq_xyz_conds[time_step] for time_step in time_steps], axis=2) * multi) / float(time_step / interval) 
    
    return seq_pose_conds, seq_xyz_conds

sceneLoadTypeCallbacks = {
    "ZJU_MoCap" : readZJUMoCapInfo,
    "DNARendering": readDNARenderingInfo,
    "I3DHuman": readID3HumanInfo,
}
