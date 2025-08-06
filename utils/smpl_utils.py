import torch
import numpy as np
import pickle

def dict_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, np.ndarray):
        return torch.from_numpy(obj).to(device)
    elif isinstance(obj, dict):
        return {k: dict_to_device(v, device) for k, v in obj.items()}
    else:
        return obj  # 其他类型保持不变

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        return u.load()
    
def SMPL_to_tensor(params, device):
    key_ = ['v_template', 'shapedirs', 'J_regressor', 'kintree_table', 'f', 'weights', "posedirs"]
    for key1 in key_:
        if key1 == 'J_regressor':
            if isinstance(params[key1], np.ndarray):
                params[key1] = torch.tensor(params[key1].astype(float), dtype=torch.float32, device=device)
            else:
                params[key1] = torch.tensor(params[key1].toarray().astype(float), dtype=torch.float32, device=device)
        elif key1 == 'kintree_table' or key1 == 'f':
            params[key1] = torch.tensor(np.array(params[key1]).astype(float), dtype=torch.long, device=device)
        else:
            params[key1] = torch.tensor(np.array(params[key1]).astype(float), dtype=torch.float32, device=device)
    return params

def batch_rodrigues_torch(poses):
    """ poses: N x 3
    """
    batch_size = poses.shape[0]
    angle = torch.norm(poses + 1e-8, p=2, dim=1, keepdim=True)
    rot_dir = poses / angle

    cos = torch.cos(angle)[:, None]
    sin = torch.sin(angle)[:, None]

    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros((batch_size, 1), device=poses.device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1)
    K = K.reshape([batch_size, 3, 3])

    ident = torch.eye(3)[None].to(poses.device)
    rot_mat = ident + sin * K + (1 - cos) * torch.matmul(K, K)

    return rot_mat

def get_rigid_transformation_torch(rot_mats, joints, parents):
    """
    rot_mats: bs x 24 x 3 x 3
    joints: bs x 24 x 3
    parents: 24
    """
    # obtain the relative joints
    bs, joints_num = joints.shape[0:2]
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    # create the transformation matrix
    transforms_mat = torch.cat([rot_mats, rel_joints[..., None]], dim=-1)
    padding = torch.zeros([bs, joints_num, 1, 4], device=rot_mats.device)  #.to(rot_mats.device)
    padding[..., 3] = 1
    transforms_mat = torch.cat([transforms_mat, padding], dim=-2)

    # rotate each part
    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:, i])
        transform_chain.append(curr_res)
    transforms = torch.stack(transform_chain, dim=1)

    # obtain the rigid transformation
    padding = torch.zeros([bs, joints_num, 1], device=rot_mats.device)  #.to(rot_mats.device)
    joints_homogen = torch.cat([joints, padding], dim=-1)
    rel_joints = torch.sum(transforms * joints_homogen[:, :, None], dim=3)
    transforms[..., 3] = transforms[..., 3] - rel_joints

    return transforms

def get_transform_params_torch(smpl, params, rot_mats=None, correct_Rs=None):
    """ obtain the transformation parameters for linear blend skinning
    """
    v_template = smpl['v_template']

    # add shape blend shapes
    shapedirs = smpl['shapedirs']
    betas = params['shapes']
    bs = betas.shape[0]
    v_template = v_template.unsqueeze(0).expand(bs, *v_template.shape)
    shapedirs = shapedirs.unsqueeze(0).expand(bs, *shapedirs.shape)
    v_shaped = v_template + torch.sum(shapedirs[...,:betas.shape[-1]] * betas[:,None,None], axis=-1).float()

    if rot_mats is None:
        # add pose blend shapes
        poses = params['poses'].reshape(-1, 3)
        # bs x 24 x 3 x 3
        rot_mats = batch_rodrigues_torch(poses).view(params['poses'].shape[0], -1, 3, 3)

        if correct_Rs is not None:
            rot_mats_no_root = rot_mats[:, 1:]
            rot_mats_no_root = torch.matmul(rot_mats_no_root.reshape(-1, 3, 3), correct_Rs.reshape(-1, 3, 3)).reshape(-1, rot_mats.shape[1]-1, 3, 3)
            rot_mats = torch.cat([rot_mats[:, 0:1], rot_mats_no_root], dim=1)

    # obtain the joints
    joints = torch.matmul(smpl['J_regressor'][None], v_shaped) # [bs, 24 ,3]

    # obtain the rigid transformation
    parents = smpl['kintree_table'][0]
    A = get_rigid_transformation_torch(rot_mats, joints, parents)

    # apply global transformation
    R = params['R'] 
    Th = params['Th'] 

    return A, R, Th, joints

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def quaternion_multiply(r, s):
    r0, r1, r2, r3 = r.unbind(-1)
    s0, s1, s2, s3 = s.unbind(-1)
    t0 = r0 * s0 - r1 * s1 - r2 * s2 - r3 * s3
    t1 = r0 * s1 + r1 * s0 - r2 * s3 + r3 * s2
    t2 = r0 * s2 + r1 * s3 + r2 * s0 - r3 * s1
    t3 = r0 * s3 - r1 * s2 + r2 * s1 + r3 * s0
    t = torch.stack([t0, t1, t2, t3], dim=-1)
    return t

def create_canonical_vertices(smpl_model, model_type='smpl'):
    """
    Generates vertices for a canonical 'A-pose' SMPL or SMPLX model.

    This function initializes SMPL/SMPLX parameters to create a model in a
    standardized A-pose (legs and arms slightly apart) with an average shape,
    no global rotation, and no translation.

    Args:
        smpl_model: The pre-loaded SMPL or SMPLX model object.
                    - For 'smpl', this model should be numpy-based.
                    - For 'smplx', this model should be torch-based.
        model_type (str): The type of the model, either 'smpl' or 'smplx'.
                          This determines which parameter structure and
                          API to use.

    Returns:
        tuple:
            - canon_params (dict): A dictionary containing the model
              parameters used to generate the vertices.
            - canon_vertices (np.ndarray): A numpy array of shape (N, 3)
              containing the 3D vertex coordinates of the canonical model.
    """
    if model_type not in ['smpl', 'smplx']:
        raise ValueError("model_type must be either 'smpl' or 'smplx'")

    # --- Common A-pose Angles (in radians) ---
    left_hip_angle = 45 / 180 * np.pi
    right_hip_angle = -45 / 180 * np.pi
    left_shoulder_angle = -30 / 180 * np.pi
    right_shoulder_angle = 30 / 180 * np.pi

    if model_type == 'smpl':
        # --- Logic for the first dataset (numpy-based SMPL) ---
        canon_params = {}
        # No global rotation or translation
        canon_params['R'] = np.eye(3, dtype=np.float32)
        canon_params['Th'] = np.zeros((1, 3), dtype=np.float32)
        # Average body shape
        canon_params['shapes'] = np.zeros((1, 10), dtype=np.float32)
        # Start with T-Pose and modify for A-pose
        canon_params['poses'] = np.zeros((1, 72), dtype=np.float32)

        # NOTE: Indices are for the 72-element full pose vector
        # Left Hip (Joint 1), Right Hip (Joint 2)
        canon_params['poses'][0, 5] = left_hip_angle
        canon_params['poses'][0, 8] = right_hip_angle
        # Left Shoulder (Joint 7), Right Shoulder (Joint 8)
        canon_params['poses'][0, 23] = left_shoulder_angle
        canon_params['poses'][0, 26] = right_shoulder_angle

        # The model call expects (poses, shapes)
        canon_vertices, _ = smpl_model(canon_params['poses'], canon_params['shapes'].reshape(-1))
        
        # Apply global orientation externally, as per the original code
        canon_vertices = (np.matmul(canon_vertices, canon_params['R'].transpose()) + canon_params['Th']).astype(np.float32)
        
        return canon_params, canon_vertices

    elif model_type == 'smplx':
        # --- Logic for the second dataset (torch-based SMPLX) ---
        canon_params = {}
        # Initialize all SMPLX parameters as numpy arrays first
        canon_params['R'], canon_params['Th'] = np.eye(3, dtype=np.float32), np.zeros((1, 3), dtype=np.float32)
        canon_params['global_orient'] = np.zeros((1, 3), dtype=np.float32)
        canon_params['transl'] = np.zeros((1, 3), dtype=np.float32)
        canon_params['betas'] = np.zeros((1, 10), dtype=np.float32)
        canon_params['expression'] = np.zeros((1, 10), dtype=np.float32)
        canon_params['body_pose'] = np.zeros((1, 63), dtype=np.float32) # 21 joints * 3
        canon_params['jaw_pose'] = np.zeros((1, 3), dtype=np.float32)
        canon_params['left_hand_pose'] = np.zeros((1, 45), dtype=np.float32)
        canon_params['right_hand_pose'] = np.zeros((1, 45), dtype=np.float32)
        canon_params['leye_pose'] = np.zeros((1, 3), dtype=np.float32)
        canon_params['reye_pose'] = np.zeros((1, 3), dtype=np.float32)

        # NOTE: Indices are for the 63-element body_pose vector (excludes global_orient)
        # Left Hip (Joint 1), Right Hip (Joint 2)
        canon_params['body_pose'][0, 2] = left_hip_angle
        canon_params['body_pose'][0, 5] = right_hip_angle
        # Left Shoulder (Joint 7), Right Shoulder (Joint 8)
        canon_params['body_pose'][0, 20] = left_shoulder_angle
        canon_params['body_pose'][0, 23] = right_shoulder_angle
        
        # Convert numpy params to torch tensors for the model
        canon_params_tensor = {key: torch.from_numpy(val) for key, val in canon_params.items()}

        # Call the SMPLX model with keyword arguments
        body_model_output = smpl_model(
            global_orient=canon_params_tensor['global_orient'],
            betas=canon_params_tensor['betas'],
            body_pose=canon_params_tensor['body_pose'],
            jaw_pose=canon_params_tensor['jaw_pose'],
            left_hand_pose=canon_params_tensor['left_hand_pose'],
            right_hand_pose=canon_params_tensor['right_hand_pose'],
            leye_pose=canon_params_tensor['leye_pose'],
            reye_pose=canon_params_tensor['reye_pose'],
            expression=canon_params_tensor['expression'],
            transl=canon_params_tensor['transl'],
            return_full_pose=True,
        )

        # Extract vertices and convert back to numpy
        canon_vertices = body_model_output.vertices.detach().numpy().astype(np.float32).reshape(-1, 3)
        
        # For consistency, add the full pose and combined shapes to the returned params dict
        canon_params['poses'] = body_model_output.full_pose.detach().numpy()
        canon_params['shapes'] = np.concatenate([canon_params['betas'], canon_params['expression']], axis=-1)

        return canon_params, canon_vertices