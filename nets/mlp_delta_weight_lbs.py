import torch
import torch.nn as nn
import torch.nn.functional as F

class LBSOffsetDecoder(nn.Module):
    def __init__(self, total_bones=24):
        super(LBSOffsetDecoder, self).__init__()

        self.total_bones = total_bones

        self.actvn = nn.ReLU()

        input_ch = 63
        D = 4
        W = 128
        self.skips = [2]
        self.bw_linears = nn.ModuleList([nn.Conv1d(input_ch, W, 1)] + [
            nn.Conv1d(W, W, 1) if i not in
            self.skips else nn.Conv1d(W + input_ch, W, 1) for i in range(D - 1)
        ])
        self.bw_fc = nn.Conv1d(W, total_bones, 1)

    def forward(self, pts_feats):
        pts_feats = pts_feats.permute(0, 2, 1)
        net = pts_feats
        for i, l in enumerate(self.bw_linears):
            net = self.actvn(self.bw_linears[i](net))
            if i in self.skips:
                net = torch.cat((pts_feats, net), dim=1)
        bw = self.bw_fc(net)   
        return bw 