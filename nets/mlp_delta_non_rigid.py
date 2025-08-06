import torch
import torch.nn as nn

class NonrigidDeformer(nn.Module):
    def __init__(self, D=3, W=512, use_pose_cond=0, use_seq_pose_cond=0, use_seq_xyz_cond=0, 
                 pos_input_dim=63, pose_cond_dim=32, seq_pose_cond_dim=32, seq_xyz_cond_dim=96,
                 seq_len=6, seq_xyz_knn=1, time_step_num=1, smpl_type='smpl'):
        super(NonrigidDeformer, self).__init__()

        self.use_pose_cond = use_pose_cond
        self.use_seq_pose_cond = use_seq_pose_cond
        self.use_seq_xyz_cond = use_seq_xyz_cond

        self.input_ch = pos_input_dim
        self.pose_cond_dim, self.seq_pose_cond_dim, self.seq_xyz_cond_dim = 0, 0, 0

        if self.use_pose_cond:
            self.PoseEncoder = PoseEncoder(32, pose_cond_dim, smpl_type)
            self.input_ch += pose_cond_dim
            
        if self.use_seq_pose_cond:
            self.SeqPoseEncoder = SeqPoseEncoder(seq_len, 16, seq_pose_cond_dim, time_step_num, smpl_type)
            self.input_ch += seq_pose_cond_dim

        if self.use_seq_xyz_cond:
            self.SeqXYZEncoder = SeqXYZEncoder(pos_emb_dim=pos_input_dim, hidden_dim1=96, hidden_dim2=256, output_dim=seq_xyz_cond_dim, 
                                        time_step_num=time_step_num, seq_len=seq_len, seq_xyz_knn=seq_xyz_knn)
            self.input_ch += seq_xyz_cond_dim
        
        layers = []
        in_dim = self.input_ch
        for _ in range(D):
            layers.append(nn.Linear(in_dim, W))
            layers.append(nn.ReLU())
            in_dim = W
        self.mlp = nn.Sequential(*layers)

        self.gaussian_warp = nn.Linear(W, 3)
        self.gaussian_rotation = nn.Linear(W, 4)
        self.gaussian_scaling = nn.Linear(W, 3)

    def forward(self, x_emb, pose_conds=None, seq_pose_conds=None, seq_xyz_conds=None):
        feats = []
        feats.append(x_emb)

        # single frame pose condition
        if self.use_pose_cond: 
            pose_feats = self.PoseEncoder(pose_conds)
            pose_feats = pose_feats.unsqueeze(1).expand(-1, x_emb.shape[1], -1)
            feats.append(pose_feats)
        
        # sequential pose condition
        if self.use_seq_pose_cond:
            seq_pose_feats = self.SeqPoseEncoder(seq_pose_conds)
            seq_pose_feats = seq_pose_feats.unsqueeze(1).expand(-1, x_emb.shape[1], -1)
            feats.append(seq_pose_feats)
        
        # sequential point-wise delta xyz condition
        if self.use_seq_xyz_cond: 
            seq_xyz_feats = self.SeqXYZEncoder(seq_xyz_conds, x_emb)
            feats.append(seq_xyz_feats)
        
        h = self.mlp(torch.cat(feats, dim=-1))
        d_xyz, d_scaling, d_rotation = self.gaussian_warp(h), self.gaussian_scaling(h), self.gaussian_rotation(h)
        
        return d_xyz, d_rotation, d_scaling

N_JOINT = {'smpl': 23, 'smplx': 54}

class PoseEncoder(nn.Module):
    def __init__(self, D1, D2, smpl_type):
        super(PoseEncoder, self).__init__()
        
        self.input_dim = 3 * N_JOINT[smpl_type] # axis-angle form
        self.mlp = nn.Sequential(nn.Linear(self.input_dim,D1), nn.ReLU(),
                                 nn.Linear(D1, D2), nn.ReLU())
    def forward(self, x):
        '''
        x: (B,J,3) Axis Angele
        return output (N,self.output_dim)
        '''
        bs = x.shape[0]
        x_joint_flat = x.view(bs, -1)

        return self.mlp(x_joint_flat)

class SeqPoseEncoder(nn.Module):
    def __init__(self, length, D1, D2, time_step_num, smpl_type):
        super(SeqPoseEncoder, self).__init__()

        self.input_dim = 3 * (N_JOINT[smpl_type] + 1) # axis-angle form, + global orientation
        self.mlp1 = nn.Sequential(nn.Linear(self.input_dim*time_step_num,D1), nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Linear(D1*length, D2), nn.ReLU())

    def forward(self, x):
        # x: (B, N, T, J, DeltaStep, C)

        bs, T = x.shape[0], x.shape[1]
        x = self.mlp1(x.view(bs, T, -1))
        x = self.mlp2(x.view(bs, -1))

        return x

class SeqXYZEncoder(nn.Module):
    def __init__(self, vel_dim=3, pos_emb_dim=63, vel_emb_dim=64, pos_emb_proj_dim=32, 
                 hidden_dim1=96, hidden_dim2=256, output_dim=128, 
                 time_step_num=1, seq_len=6, seq_xyz_knn=5):
        super(SeqXYZEncoder, self).__init__()

        self.vel_encoder = nn.Sequential(nn.Linear(vel_dim*seq_xyz_knn*time_step_num, vel_emb_dim), nn.ReLU())
        self.pos_emb_proj = nn.Sequential(nn.Linear(pos_emb_dim, pos_emb_proj_dim), nn.ReLU())
        
        self.mlp1 = nn.Sequential(nn.Linear(vel_emb_dim+pos_emb_proj_dim, hidden_dim1), nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Linear(hidden_dim1*seq_len, hidden_dim2), nn.ReLU(),
                                  nn.Linear(hidden_dim2, output_dim), nn.ReLU())
    def forward(self, x, x_emb):
        # x -> B, N, T, KNN, DeltaStep, C
        B, N, T = x.shape[0], x.shape[1], x.shape[2]

        pos_feat = self.pos_emb_proj(x_emb)
        pos_feat = pos_feat.unsqueeze(2).expand(-1, -1, T, -1)
        vel_emb = self.vel_encoder(x.view(B, N, T, -1))

        h = torch.concat([vel_emb, pos_feat], dim=-1)
        h = self.mlp1(h)
        h = self.mlp2(h.view(B, N, -1))

        return h