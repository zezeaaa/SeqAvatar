import torch
import os
import cv2
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
import multiprocessing
from smpl_model.smplx.body_models import SMPLX
from utils.SMCReader import SMCReader

# global
g_smc_reader = None
g_smc_annots_reader = None
g_output_path = None


def init_worker(reader, annots_reader, out_path):
    global g_smc_reader, g_smc_annots_reader, g_output_path
    g_smc_reader = reader
    g_smc_annots_reader = annots_reader
    g_output_path = out_path


def process_view_task(task_params):
    pose_index, view_index, image_scaling, white_background = task_params
    try:
        cam_params = g_smc_annots_reader.get_Calibration(view_index)
        if view_index < 48:
            image = g_smc_reader.get_img('Camera_5mp', view_index, Image_type='color', Frame_id=pose_index)
        else:
            image = g_smc_reader.get_img('Camera_12mp', view_index, Image_type='color', Frame_id=pose_index)
        
        bkgd_mask = g_smc_annots_reader.get_mask(view_index, Frame_id=pose_index)
        bkgd_mask[bkgd_mask != 0] = 255
        bkgd_mask = bkgd_mask / 255.0

        # undistort
        K = cam_params['K']
        D = cam_params['D']

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        image = cv2.undistort(image, K, D)
        bkgd_mask = cv2.undistort(bkgd_mask, K, D)
        image[bkgd_mask == 0] = 1 if white_background else 0

        if image_scaling != 1.:
            H, W = int(image.shape[0] * image_scaling), int(image.shape[1] * image_scaling)
            image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
            bkgd_mask = cv2.resize(bkgd_mask, (W, H), interpolation=cv2.INTER_NEAREST)
            K[:2] = K[:2] * image_scaling
            cam_params['K'] = K
        
        # save image
        img_path = os.path.join(g_output_path, 'images', f'{view_index:02d}', f'{pose_index:06d}.png')
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        image = Image.fromarray((image * 255.0).astype(np.uint8), "RGB")
        image.save(img_path)

        # save mask
        bkgd_mask_path = os.path.join(g_output_path, 'bkgd_masks', f'{view_index:02d}', f'{pose_index:06d}.png')
        os.makedirs(os.path.dirname(bkgd_mask_path), exist_ok=True)
        bkgd_mask = Image.fromarray((bkgd_mask * 255.0).astype(np.uint8))
        bkgd_mask.save(bkgd_mask_path)

        # save camera
        cam_path = os.path.join(g_output_path, 'cameras', f'{view_index:02d}', f'{pose_index:06d}.npz')
        os.makedirs(os.path.dirname(cam_path), exist_ok=True)
        np.savez(cam_path, **cam_params)

        return f"Success: Pose {pose_index}, View {view_index}"
    except Exception as e:
        return f"Error on Pose {pose_index}, View {view_index}: {e}"


def get_models(pose_index, smc_annots_reader, smpl_model):        
    smpl_dict = smc_annots_reader.get_SMPLx(Frame_id=pose_index)
    smpl_data = {}
    smpl_data['global_orient'] = smpl_dict['fullpose'][0].reshape(-1)
    smpl_data['body_pose'] = smpl_dict['fullpose'][1:22].reshape(-1)
    smpl_data['jaw_pose'] = smpl_dict['fullpose'][22].reshape(-1)
    smpl_data['leye_pose'] = smpl_dict['fullpose'][23].reshape(-1)
    smpl_data['reye_pose'] = smpl_dict['fullpose'][24].reshape(-1)
    smpl_data['left_hand_pose'] = smpl_dict['fullpose'][25:40].reshape(-1)
    smpl_data['right_hand_pose'] = smpl_dict['fullpose'][40:55].reshape(-1)
    smpl_data['transl'] = smpl_dict['transl'].reshape(-1)
    smpl_data['betas'] = smpl_dict['betas'].reshape(-1)
    smpl_data['expression'] = smpl_dict['expression'].reshape(-1)
    
    # load smpl data
    smpl_param = {
        'global_orient': np.expand_dims(smpl_data['global_orient'].astype(np.float32), axis=0),
        'transl': np.expand_dims(smpl_data['transl'].astype(np.float32), axis=0),
        'body_pose': np.expand_dims(smpl_data['body_pose'].astype(np.float32), axis=0),
        'jaw_pose': np.expand_dims(smpl_data['jaw_pose'].astype(np.float32), axis=0),
        'betas': np.expand_dims(smpl_data['betas'].astype(np.float32), axis=0),
        'expression': np.expand_dims(smpl_data['expression'].astype(np.float32), axis=0),
        'leye_pose': np.expand_dims(smpl_data['leye_pose'].astype(np.float32), axis=0),
        'reye_pose': np.expand_dims(smpl_data['reye_pose'].astype(np.float32), axis=0),
        'left_hand_pose': np.expand_dims(smpl_data['left_hand_pose'].astype(np.float32), axis=0),
        'right_hand_pose': np.expand_dims(smpl_data['right_hand_pose'].astype(np.float32), axis=0),
    }

    smpl_param['R'] = np.eye(3, dtype=np.float32) 
    smpl_param['Th'] = smpl_param['transl'].astype(np.float32)
    
    smpl_param_tensor = {}
    for key in smpl_param.keys():
        smpl_param_tensor[key] = torch.from_numpy(smpl_param[key])
    
    body_model_output = smpl_model(
        global_orient=smpl_param_tensor['global_orient'],
        betas=smpl_param_tensor['betas'],
        body_pose=smpl_param_tensor['body_pose'],
        jaw_pose=smpl_param_tensor['jaw_pose'],
        left_hand_pose=smpl_param_tensor['left_hand_pose'],
        right_hand_pose=smpl_param_tensor['right_hand_pose'],
        leye_pose=smpl_param_tensor['leye_pose'],
        reye_pose=smpl_param_tensor['reye_pose'],
        expression=smpl_param_tensor['expression'],
        transl=smpl_param_tensor['transl'],
        return_full_pose=True,
    )
    
    smpl_param['poses'] = body_model_output.full_pose.detach().cpu().numpy().astype(np.float32)
    smpl_param['shapes'] = np.concatenate([smpl_param['betas'], smpl_param['expression']], axis=-1, dtype=np.float32)
    smpl_param['obs_xyz'] = body_model_output.vertices.detach().numpy().astype(np.float32).reshape(-1, 3)
    return smpl_param

if __name__ == "__main__":
    if multiprocessing.get_start_method(allow_none=True) != 'fork':
        multiprocessing.set_start_method('fork', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dna_root', type=str, default='mydata/DNARendering_process/')
    parser.add_argument('--output_root', type=str, default='mydata/DNARendering_process6/')
    parser.add_argument('--white_background', action='store_true')
    parser.add_argument('--scenes', nargs='+', default=['0007_04', '0019_10', '0044_11', '0051_09', '0206_04', '0813_05'])
    args = parser.parse_args()

    for scene_name in args.scenes:
        scene_path = os.path.join(args.dna_root, scene_name)
        output_path = os.path.join(args.output_root, scene_name)
        
        main_path = os.path.join(scene_path, scene_name + '.smc')
        annots_file_path = main_path.replace('main', 'annotations').split('.')[0] + '_annots.smc'

        smc_reader = SMCReader(main_path)
        smc_annots_reader = SMCReader(annots_file_path)
        gender = smc_reader.actor_info['gender']
        smpl_model = SMPLX('smpl_model/models/', smpl_type='smplx', gender=gender, 
                        use_face_contour=True, flat_hand_mean=False, use_pca=False,
                        num_pca_comps=24, num_betas=10, num_expression_coeffs=10, ext='npz')

        pose_start, pose_end = 0, 150
        views = [i for i in range(0, 48, 2)] + [i for i in range(48, 60, 2)]
        image_scaling = 0.25

        with multiprocessing.Pool(processes=None, initializer=init_worker, initargs=(smc_reader, smc_annots_reader, output_path)) as pool:
            for pose_index in tqdm(range(pose_start, pose_end), desc=f"Processing {scene_name}"):
                # save SMPL model                
                model_path = os.path.join(output_path, 'model', f'{pose_index:06d}.npz')
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                smpl_param = get_models(pose_index, smc_annots_reader, smpl_model)
                np.savez(model_path, **smpl_param)  

                # save each view's camera, image, mask
                views_for_current_pose = [(pose_index, view, image_scaling, args.white_background) for view in views]
                pool.map(process_view_task, views_for_current_pose)
        
        print(f'\nFinsh processing {scene_name}!')
