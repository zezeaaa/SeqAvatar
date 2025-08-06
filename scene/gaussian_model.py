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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, get_embedder
from knn_cuda import KNN

import torch.nn.functional as F
from nets.mlp_delta_body_pose import BodyPoseRefiner
from nets.mlp_delta_weight_lbs import LBSOffsetDecoder
from nets.mlp_delta_non_rigid import NonrigidDeformer
from utils.smpl_utils import read_pickle, SMPL_to_tensor, get_transform_params_torch, batch_rodrigues, quaternion_multiply

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation, transform):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            if transform is not None:
                actual_covariance = transform @ actual_covariance
                actual_covariance = actual_covariance @ transform.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int, smpl_type : str, motion_offset_flag : bool, actor_gender: str, args):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.mlp_optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.device=torch.device('cuda', torch.cuda.current_device())
        # load SMPL model
        if smpl_type == 'smpl':
            neutral_smpl_path = os.path.join('smpl_model/models', f'SMPL_{actor_gender.upper()}.pkl')
        elif smpl_type == 'smplx':
            neutral_smpl_path = os.path.join('smpl_model/models', f'SMPLX_{actor_gender.upper()}.pkl')
        self.SMPL_NEUTRAL = SMPL_to_tensor(read_pickle(neutral_smpl_path), device=self.device)
        self.canon_params = None
        self.canon_vertices = None
        self.A2T_pose_tranform = None
        self.canon_pose_offsets = None
        self.smpl_params_dict = None
        self.cond_dict = None

        # load knn module
        self.knn = KNN(k=1, transpose_mode=True)
        self.seq_xyz_knn = args.seq_xyz_knn
        self.custom_knn_near = KNN(k=self.seq_xyz_knn, transpose_mode=True)

        self.motion_offset_flag = motion_offset_flag
        self.non_rigid_flag = args.non_rigid_flag
        self.nonrigid_poseconds_flag = args.nonrigid_poseconds_flag
        self.nonrigid_deltaposeconds_flag = args.nonrigid_deltaposeconds_flag
        self.nonrigid_deltaxyzconds_flag = args.nonrigid_deltaxyzconds_flag

        if self.motion_offset_flag:
            # load pose correction module
            total_bones = self.SMPL_NEUTRAL['weights'].shape[-1]
            self.pose_decoder = BodyPoseRefiner(total_bones=total_bones, embedding_size=3*(total_bones-1), mlp_width=128, mlp_depth=2).to(self.device)

            # load lbs weight module
            self.pos_embed_fn, pos_embed_ch = get_embedder(10, 3)
            self.lweight_offset_decoder = LBSOffsetDecoder(total_bones).to(self.device)

            # non-rigid deformer
            if self.non_rigid_flag:
                self.non_rigid_deformer = NonrigidDeformer(pos_input_dim=pos_embed_ch,
                        use_pose_cond=self.nonrigid_poseconds_flag, use_seq_pose_cond=self.nonrigid_deltaposeconds_flag, use_seq_xyz_cond=self.nonrigid_deltaxyzconds_flag, 
                        seq_len=args.seq_len, seq_xyz_knn=self.seq_xyz_knn, time_step_num=args.time_step_num, smpl_type=smpl_type).to(self.device)
                            
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.pose_decoder,
            self.lweight_offset_decoder,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self.pose_decoder,
        self.lweight_offset_decoder) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1, transform=None, d_rotation=None, d_scaling=None):
        if d_rotation is not None:
            scaling = self.get_scaling + d_scaling
            # rotation = self._rotation + d_rotation

            q1 = d_rotation
            # q1[0] = 1. # [1,0,0,0] represents identity rotation
            # d_rotation = d_rotation[1:]
            q2 = self._rotation
            rotation = quaternion_multiply(q1, q2)

        else:
            scaling = self.get_scaling
            rotation = self._rotation
        return self.covariance_activation(scaling, scaling_modifier, rotation, transform)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        mlp_l = []
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        if self.motion_offset_flag:
            mlp_l = [
                {'params': self.pose_decoder.parameters(), 'lr': training_args.pose_refine_lr, "name": "pose_decoder"},
                {'params': self.lweight_offset_decoder.parameters(), 'lr': training_args.lbs_offset_lr, "name": "lweight_offset_decoder"},
            ]

        if self.non_rigid_flag:
            mlp_l += [{'params': self.non_rigid_deformer.parameters(), 'lr': training_args.non_rigid_deformer_lr,
                 "name": "non_rigid_deformer"},]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.mlp_optimizer = torch.optim.Adam(params=mlp_l, lr=0.001, eps=1e-15)
        gamma = training_args.mlp_lr_ratio ** (1. / training_args.iterations)
        self.mlp_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.mlp_optimizer, gamma=gamma)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # if len(group["params"]) == 1:
            if group["name"] in ['xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation']:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # assert len(group["params"]) == 1
            if group["name"] in ['xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation']:
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        self.prune_points(prune_mask)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def get_canon2Tpose_transform(self, cannon_pose_params):
        self.A2T_pose_tranform, _, _, _ = get_transform_params_torch(self.SMPL_NEUTRAL, cannon_pose_params)
        # pose dir
        vertices_num = self.canon_vertices.shape[1]
        posedirs = self.SMPL_NEUTRAL['posedirs'].cuda().float()
        pose_ = cannon_pose_params['poses']
        ident = torch.eye(3).cuda().float()
        batch_size = pose_.shape[0]
        rot_mats = batch_rodrigues(pose_.view(-1, 3)).view([batch_size, -1, 3, 3])
        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])#.cuda()
        
        self.canon_pose_offsets = torch.matmul(pose_feature.unsqueeze(1), posedirs.view(vertices_num*3, -1).transpose(1,0).unsqueeze(0)).view(batch_size, -1, 3)

    def coarse_deform_c2source(self, query_pts, params, lbs_weights=None, correct_Rs=None, return_transl=False):
        bs = query_pts.shape[0]
        joints_num = self.SMPL_NEUTRAL['weights'].shape[-1]
        vertices_num = self.canon_vertices.shape[1]
        # Find nearest smpl vertex        
        _, vert_ids = self.knn(self.canon_vertices, query_pts)
        if lbs_weights is None:
            bweights = self.SMPL_NEUTRAL['weights'][vert_ids].view(*vert_ids.shape[:2], joints_num)#.cuda() # [bs, points_num, joints_num]
        else:
            bweights = self.SMPL_NEUTRAL['weights'][vert_ids].view(*vert_ids.shape[:2], joints_num)
            bweights = torch.log(bweights + 1e-9) + lbs_weights
            bweights = F.softmax(bweights, dim=-1)

        ### From A Pose (canonical space) To T Pose
        A2T_pose_RT = torch.matmul(bweights, self.A2T_pose_tranform.reshape(bs, joints_num, -1))
        A2T_pose_RT = torch.reshape(A2T_pose_RT, (bs, -1, 4, 4))
        query_pts = query_pts - A2T_pose_RT[..., :3, 3]
        A2T_pose_R_inv = torch.inverse(A2T_pose_RT[..., :3, :3].float())
        query_pts = torch.matmul(A2T_pose_R_inv, query_pts[..., None]).squeeze(-1)

        # transforms from A Pose (canonical space) To T Pose
        transforms = A2T_pose_R_inv
        translation = None

        canon_pose_offsets = torch.gather(self.canon_pose_offsets, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
        query_pts = query_pts - canon_pose_offsets

        # From mean shape to normal shape
        shapedirs = self.SMPL_NEUTRAL['shapedirs'][..., :params['shapes'].shape[-1]]#.cuda()
        shapedirs = shapedirs.unsqueeze(0).expand(bs, *shapedirs.shape)
        shape_offset = torch.matmul(shapedirs, torch.reshape(params['shapes'].cuda(), (bs, 1, -1, 1))).squeeze(-1)
        shape_offset = torch.gather(shape_offset, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
        query_pts = query_pts + shape_offset

        posedirs = self.SMPL_NEUTRAL['posedirs']#.cuda().float()
        ident = torch.eye(3).cuda().float()
        rot_mats = params['rot_mats']

        if correct_Rs is not None:
            rot_mats_no_root = rot_mats[:, 1:]
            rot_mats_no_root = torch.matmul(rot_mats_no_root, correct_Rs)
            rot_mats = torch.cat([rot_mats[:, 0:1], rot_mats_no_root], dim=1)

        tgt_pose_feature = (rot_mats[:, 1:, :, :] - ident).view([bs, -1])#.cuda()
        tgt_pose_offsets = torch.matmul(tgt_pose_feature.unsqueeze(1), posedirs.view(vertices_num*3, -1).transpose(1,0).unsqueeze(0)).view(bs, -1, 3)
        tgt_pose_offsets = torch.gather(tgt_pose_offsets, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
        query_pts = query_pts + tgt_pose_offsets

        # T Pose to target Pose (observation space)
        cnt2tgt_rigid_RT, global_R, global_Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, params, rot_mats=rot_mats)
        cnt2tgt_rigid_RT = torch.matmul(bweights, cnt2tgt_rigid_RT.reshape(bs, joints_num, -1))
        cnt2tgt_rigid_RT = torch.reshape(cnt2tgt_rigid_RT, (bs, -1, 4, 4))
        smpl_tgt_pts = torch.matmul(cnt2tgt_rigid_RT[..., :3, :3], query_pts[..., None]).squeeze(-1)
        smpl_tgt_pts = smpl_tgt_pts + cnt2tgt_rigid_RT[..., :3, 3]
        transforms = torch.matmul(cnt2tgt_rigid_RT[..., :3, :3], transforms)

        # transform points from the smpl space to the world space
        global_R_inv = torch.inverse(global_R)
        world_pts = torch.matmul(smpl_tgt_pts, global_R_inv) + global_Th.view(bs, 1, -1)
        transforms = torch.matmul(global_R.view(bs, 1, 3, 3), transforms)

        # all transl
        if return_transl: 
            translation = -A2T_pose_RT[..., :3, 3]
            translation = torch.matmul(A2T_pose_R_inv, translation[..., None]).squeeze(-1)
            translation = translation - canon_pose_offsets + shape_offset + tgt_pose_offsets
            translation = torch.matmul(cnt2tgt_rigid_RT[..., :3, :3], translation[..., None]).squeeze(-1) + cnt2tgt_rigid_RT[..., :3, 3]
            translation = torch.matmul(translation, global_R_inv).squeeze(-1) + global_Th
        
        return world_pts, transforms, translation
