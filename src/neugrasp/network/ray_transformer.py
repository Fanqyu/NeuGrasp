import sys

import numpy as np
from einops import (rearrange, repeat)

sys.path.append('/path/to/NeuGrasp')

from src.neugrasp.network.grid_sample import grid_sample_3d
from src.neugrasp.network.LoFTR import LocalFeatureTransformer
from src.neugrasp.network.neus import *

import math

PI = math.pi


class PositionEncoding(nn.Module):
    def __init__(self, L=10):
        super().__init__()
        self.L = L
        self.augmented = rearrange((PI * 2 ** torch.arange(-1, self.L - 1)), "L -> L 1 1 1")

    def forward(self, x):
        sin_term = torch.sin(self.augmented.type_as(x) * rearrange(x, "RN SN Dim -> 1 RN SN Dim"))  # BUG?
        cos_term = torch.cos(self.augmented.type_as(x) * rearrange(x, "RN SN Dim -> 1 RN SN Dim"))
        sin_cos_term = torch.stack([sin_term, cos_term])

        sin_cos_term = rearrange(sin_cos_term, "Num2 L RN SN Dim -> (RN SN) (L Num2 Dim)")

        return sin_cos_term


class RayTransformer(nn.Module):
    """
    Ray transformer
    """

    def __init__(self, cfg, img_feat_dim=32):  # NOTE
        super().__init__()

        self.cfg = cfg
        self.volume_reso = self.cfg['volume_resolution']
        self.offset = [[0, 0, 0]]

        self.only_volume = False
        if self.only_volume:
            assert self.volume_reso > 0, "if only use volume feature, must have volume"

        self.img_feat_dim = img_feat_dim
        self.fea_volume_dim = 16 if self.cfg['feature_volume_type'] == '3d_unet' else 32

        self.PE_d_hid = 8

        # transformers
        self.density_view_transformer = LocalFeatureTransformer(d_model=self.img_feat_dim + self.fea_volume_dim,
                                                                nhead=8, layer_names=['self'], attention='linear')

        self.density_bg_transformer = LocalFeatureTransformer(d_model=self.img_feat_dim + self.fea_volume_dim,
                                                              nhead=8, layer_names=['self'], attention='linear')

        self.density_ray_transformer = LocalFeatureTransformer(
            d_model=64 + self.PE_d_hid,
            nhead=8, layer_names=['self'], attention='linear')

        if self.only_volume:
            self.DensityMLP = nn.Sequential(
                nn.Linear(self.fea_volume_dim, 32), nn.ReLU(inplace=True),
                nn.Linear(32, 16), nn.ReLU(inplace=True),
                nn.Linear(16, 1))
        else:
            self.DensityMLP = nn.Sequential(
                nn.Linear(64 + self.PE_d_hid, 32), nn.ReLU(inplace=True),
                nn.Linear(32, 16), nn.ReLU(inplace=True),
                nn.Linear(16, 1))

        self.relu = nn.ReLU(inplace=True)

        # learnable view token
        self.viewToken_view = ViewTokenNetwork(dim=self.img_feat_dim + self.fea_volume_dim)  # dim = 32 + 16
        self.viewToken_bg = ViewTokenNetwork(dim=self.img_feat_dim + self.fea_volume_dim)  # dim = 32 + 16
        self.softmax = nn.Softmax(dim=-2)

        # to calculate radiance weight
        self.linear_radianceweight_1_softmax = nn.Sequential(
            nn.Linear((self.img_feat_dim + self.fea_volume_dim) * 2 + 4, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            # 4 because of dir_diff + dot product
            nn.Linear(32, 16), nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )
        self.embed_fn, input_ch = get_embedder(3, input_dims=3)

        self.fusionMLP = nn.Sequential(nn.Linear((self.img_feat_dim + self.fea_volume_dim) * 2, 80),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(80, 64),
                                       nn.ReLU(inplace=True))

        self.PositionMLP = nn.Sequential(nn.Linear(64 + input_ch, 72),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(72, 64),
                                         nn.ReLU(inplace=True))

    def order_posenc(self, d_hid, n_samples):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_samples)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.from_numpy(sinusoid_table)

        return sinusoid_table

    def forward(self, ref_rgb_feat, ref_resi_feat, dir_diff, mask, point3D, fea_volume=None, residual_volume=None):
        """
            ref_rgb_feat: [qn_rn, dn, rfn, n_rgb_feat]
            dir_diff: [qn_rn, dn, rfn, 4]   diff & dot_product
            mask: [qn_rn, dn, rfn, 1]
            point3D: [qn, rn, dn, 3]  
            feat_volume: [1, 16, Z, Y, X]
        """
        img_rgb_sampled = ref_rgb_feat[..., :3]  # [QN_RN, SN, NV, 3]
        img_feat_sampled = ref_rgb_feat[..., 3:]  # [QN_RN, SN, NV, 32]
        resi_feat_sampled = ref_resi_feat
        dir_relative = dir_diff
        mask = mask

        _, _, NV, _ = ref_rgb_feat.shape
        QN, RN, SN, _ = point3D.shape

        if self.volume_reso > 0:
            assert fea_volume is not None
            fea_volume_feat = grid_sample_3d(fea_volume, point3D.unsqueeze(1).float())
            fea_volume_feat = rearrange(fea_volume_feat, "Dim1 C RN SN -> (Dim1 RN SN) C")

            res_volume_feat = grid_sample_3d(residual_volume, point3D.unsqueeze(1).float())
            res_volume_feat = rearrange(res_volume_feat, "Dim1 C RN SN -> (Dim1 RN SN) C")

        # --------- run transformer to aggregate information
        # -- 1. view transformer
        x = rearrange(img_feat_sampled, "QN_RN SN NV C -> (QN_RN SN) NV C")
        y = rearrange(resi_feat_sampled, "QN_RN SN NV C -> (QN_RN SN) NV C")

        if self.volume_reso > 0:
            x_fea_volume_feat = repeat(fea_volume_feat, "B_RN_SN C -> B_RN_SN NV C", NV=NV)
            x = torch.cat([x, x_fea_volume_feat], dim=-1)  # axis also can work here  [B_RN_SN, NV, 32 + 16]

            y_fea_volume_feat = repeat(res_volume_feat, "B_RN_SN C -> B_RN_SN NV C", NV=NV)
            y = torch.cat([y, y_fea_volume_feat], dim=-1)  # axis also can work here  [B_RN_SN, NV, 32 + 16]
        #  NOTE: B, QN all equal to 1, but it means different

        # add additional view aggregation token
        viewdiff_token = self.viewToken_view(x)  # [QN_SN_RN, 48]
        viewdiff_token = rearrange(viewdiff_token, "QN_RN_SN C -> QN_RN_SN 1 C")
        x = torch.cat([viewdiff_token, x], dim=1)  # [QN_RN_SN, NV + 1, 32 + 16]
        x = self.density_view_transformer(x)  # [QN_RN_SN, NV + 1, 32 + 16]

        bgdiff_token = self.viewToken_bg(y)  # [QN_SN_RN, 48]
        bgdiff_token = rearrange(bgdiff_token, "QN_RN_SN C -> QN_RN_SN 1 C")
        y = torch.cat([bgdiff_token, y], dim=1)  # [QN_RN_SN, NV + 1, 32 + 16]
        y = self.density_bg_transformer(y)  # [QN_RN_SN, NV + 1, 32 + 16]

        x1 = rearrange(x, "QN_RN_SN NV C -> NV QN_RN_SN C")
        x = x1[0]  # reference      projection feature
        view_feature = x1[1:]

        y1 = rearrange(y, "QN_RN_SN NV C -> NV QN_RN_SN C")
        y = y1[0]  # reference      projection feature
        residual_feature = y1[1:]

        z = torch.cat([x, y], dim=-1)
        z = self.fusionMLP(z)

        with torch.set_grad_enabled(True):
            point3D.requires_grad_(True)
            embed_pts = self.embed_fn(point3D)[0]  # [RN, SN, 21]
            if self.only_volume:
                z = rearrange(x_fea_volume_feat, "(QN RN SN) NV C -> NV (QN RN) SN C", QN=QN, RN=RN, SN=SN)[0]
            else:
                # -- 2. ray transformer
                # add positional encoding
                z = rearrange(z, "(QN RN SN) C -> (QN RN) SN C", RN=RN, QN=QN, SN=SN)
                z = torch.cat([z, embed_pts], dim=-1)  # [QN_RN, SN, 64 + 21]
                z = self.PositionMLP(z)  # [QN_RN, SN, 64]
                z = torch.cat([z, repeat(self.order_posenc(d_hid=self.PE_d_hid, n_samples=SN).type_as(z),
                                         "SN C -> QN_RN SN C", QN_RN=QN * RN)], dim=2)
                num_valid_ref = torch.sum(mask, dim=2).squeeze(-1)
                z = self.density_ray_transformer(z, mask0=(num_valid_ref > 1).float())  # [QN_RN, SN, 64 + 8]

            sdf = self.DensityMLP(z).clip(-1., 1.)  # [QN_RN, SN, 1]  TODO: clip
            sdf = sdf.masked_fill((num_valid_ref < 1).unsqueeze(-1), 1.)  # set the sigma of invalid point to zero

            d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            gradients = torch.autograd.grad(
                outputs=sdf,
                inputs=point3D,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]  # return: tuple(tensor, ) so [0], [n_rays, n_samples, 3]

        # calculate weight using view transformers result
        view_feature = rearrange(view_feature, "NV (QN RN SN) C -> QN RN SN NV C", QN=QN, RN=RN, SN=SN)
        residual_feature = rearrange(residual_feature, "NV (QN RN SN) C -> QN RN SN NV C", QN=QN, RN=RN, SN=SN)
        dir_relative = rearrange(dir_relative, "(QN RN) SN NV Dim3 -> QN RN SN NV Dim3", QN=QN, RN=RN)

        x_weight = torch.cat([view_feature, residual_feature, dir_relative], dim=-1)  # [QN, RN, SN, NV, 48 * 2 + 4]
        x_weight = self.linear_radianceweight_1_softmax(x_weight)  # [QN, RN, SN, NV, 1]
        mask = rearrange(mask, "(QN RN) SN NV 1 -> QN RN SN NV 1", QN=QN, RN=RN)  # [QN, RN, SN, NV, 1]
        x_weight[mask == 0] = -1e9  # penalize, subsequently passing through softmax
        weight = self.softmax(x_weight)

        radiance = (img_rgb_sampled * rearrange(weight, "QN RN SN NV 1 -> (QN RN) SN NV 1")).sum(
            dim=2)  # [QN_RN, SN, 3]
        out = torch.cat([radiance, sdf], dim=-1)

        return out, gradients


class ViewTokenNetwork(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_parameter('view_token', nn.Parameter(torch.randn([1, dim])))  # [1, 48]

    def forward(self, x):
        return torch.ones([len(x), 1]).type_as(x) * self.view_token  # [BN_SN_RN, 48]
