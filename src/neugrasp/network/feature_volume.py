import sys

import numpy as np
import torch
from einops import (rearrange, repeat)
from torch import nn

sys.path.append('/path/to/NeuGrasp')

from src.neugrasp.network.grid_sample import grid_sample_2d
from src.neugrasp.network.cnn3d import Volume3DUnet, Volume3DDecoder


class FeatureVolume(nn.Module):
    """
    Create the coarse feature volume in a MVS-like way
    """

    def __init__(self, cfg):
        """
        Set up the volume grid given resolution
        Note: the xyz of original implementation utilizes normalized coords which ranges from -1 to 1. It is natural
              since the world frame of VolRecon is normalized to [-1, 1]. However, it doesn't fit for NeuGrasp, since
              the world frame here is real-world size, and [0.15, 0.15, 0.05] that is in the tsdf frame is the origin
              of the blender world frame.
        """
        super().__init__()

        self.cfg = cfg
        self.volume_reso = self.cfg['volume_resolution']
        self.volume_size = self.cfg['volume_size']
        self.voxel_size = self.volume_size / self.volume_reso
        self.half = self.voxel_size / 2
        self.volume_regularization = Volume3DUnet()

        x_line = (np.linspace(0, self.volume_reso - 1, self.volume_reso)) * self.voxel_size + self.half  # [0 ~ 39]
        y_line = (np.linspace(0, self.volume_reso - 1, self.volume_reso)) * self.voxel_size + self.half
        z_line = (np.linspace(0, self.volume_reso - 1, self.volume_reso)) * self.voxel_size + self.half

        # # create the volume grid
        self.x, self.y, self.z = np.meshgrid(x_line, y_line, z_line, indexing='ij')  # [40, 40, 40]
        self.xyz = np.stack([self.x, self.y, self.z])  # [3, 40, 40, 40]

        self.linear = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 16), nn.ReLU(inplace=True),
            nn.Linear(16, 8)
        )

    def forward(self, feats, ref_poses, Ks, bbox, img_shape):
        """
        feats: [NV C H W], NV: number of views
        """
        h, w = img_shape[-2:]
        normalizer = torch.tensor([w, h]).type_as(feats)  # NOTE
        ref_poses_h = repeat(torch.eye(4), "X Y -> NV X Y", NV=len(ref_poses)).clone().type_as(ref_poses)
        ref_poses_h[:, :3, :4] = ref_poses
        intrinsics_pad = repeat(torch.eye(4), "X Y -> NV X Y", NV=len(Ks)).clone().type_as(Ks)
        intrinsics_pad[:, :3, :3] = Ks
        w2pixel = intrinsics_pad @ ref_poses_h
        NV, _, _, _ = feats.shape
        bbox0 = torch.tensor(bbox[0]).type_as(feats)[..., None, None, None]

        # ---- step 1: projection -----------------------------------------------
        volume_xyz = torch.tensor(self.xyz).type_as(w2pixel) + bbox0
        volume_xyz = volume_xyz.reshape([3, -1])
        volume_xyz_homo = torch.cat([volume_xyz, torch.ones_like(volume_xyz[0:1])], dim=0)  # [4,XYZ]

        volume_xyz_homo_NV = repeat(volume_xyz_homo, "Num4 XYZ -> NV Num4 XYZ", NV=NV)

        # volume project into views
        volume_xyz_pixel_homo = w2pixel @ volume_xyz_homo_NV  # NV 4 4 @ NV 4 XYZ
        volume_xyz_pixel_homo = volume_xyz_pixel_homo[:, :3]
        mask_valid_depth = volume_xyz_pixel_homo[:, 2] > 0  # NV XYZ
        mask_valid_depth = mask_valid_depth.float()

        volume_xyz_pixel = volume_xyz_pixel_homo / volume_xyz_pixel_homo[:, 2:3]
        volume_xyz_pixel = volume_xyz_pixel[:, :2]
        volume_xyz_pixel = rearrange(volume_xyz_pixel, "NV Dim2 XYZ -> NV XYZ Dim2")
        volume_xyz_pixel = volume_xyz_pixel.unsqueeze(2) / normalizer - 1.  # [NV, XYZ, 1, Dim2] normalize
        # tqdm.write(f'{volume_xyz_pixel[-1]}\n{volume_xyz_pixel[:, -1]}')

        # projection: project all x * y * z points to NV images and sample features

        # grid sample 2D
        volume_feature, mask = grid_sample_2d(feats, volume_xyz_pixel)  # NV C XYZ 1, NV XYZ 1

        volume_feature = volume_feature.squeeze(-1)  # NV C XYZ
        mask = mask.squeeze(-1)  # NV XYZ
        mask = mask * mask_valid_depth

        volume_feature = rearrange(volume_feature, "NV C (NumX NumY NumZ) -> NV NumX NumY NumZ C", NV=NV,
                                   NumX=self.volume_reso, NumY=self.volume_reso, NumZ=self.volume_reso)
        mask = rearrange(mask, "NV (NumX NumY NumZ) -> NV NumX NumY NumZ", NV=NV, NumX=self.volume_reso,
                         NumY=self.volume_reso, NumZ=self.volume_reso)

        weight = mask / (torch.sum(mask, dim=1, keepdim=True) + 1e-8)
        weight = weight.unsqueeze(-1)  # NV X Y Z 1

        # ---- step 2: compress ------------------------------------------------
        volume_feature_compressed = self.linear(volume_feature)  # [NV, X, Y, Z, 8]

        # ---- step 3: mean, var ------------------------------------------------
        mean = torch.sum(volume_feature_compressed * weight, dim=0, keepdim=True)  # 1 X Y Z 8
        var = torch.sum(weight * (volume_feature_compressed - mean) ** 2, dim=0, keepdim=True)  # 1 X Y Z 8

        volume_mean_var = torch.cat([mean, var], axis=-1)  # [1 X Y Z 16]
        volume_mean_var = volume_mean_var.permute(0, 4, 3, 2, 1)  # [1,16,Z,Y,X]

        # ---- step 4: 3D regularization ----------------------------------------
        volume_mean_var_reg = self.volume_regularization(volume_mean_var)  # mean + var = 16

        return volume_mean_var_reg  # [1, 16, Z, Y, X]


class MultiScaleFeatureVolume(nn.Module):
    """
    Create the coarse feature volume in a MVS-like way
    """

    def __init__(self, cfg):
        """
        Set up the volume grid given resolution
        """
        super().__init__()

        self.cfg = cfg
        self.volume_reso = self.cfg['volume_resolution']
        self.volume_size = self.cfg['volume_size']
        self.voxel_size = self.volume_size / self.volume_reso
        self.half = self.voxel_size / 2
        self.volume_regularization = Volume3DDecoder()

        x_line = (np.linspace(0, self.volume_reso - 1, self.volume_reso)) * self.voxel_size + self.half  # [0 ~ 39]
        y_line = (np.linspace(0, self.volume_reso - 1, self.volume_reso)) * self.voxel_size + self.half
        z_line = (np.linspace(0, self.volume_reso - 1, self.volume_reso)) * self.voxel_size + self.half

        self.multilevel = 3  # NOTE
        # create the volume grid
        self.x, self.y, self.z = np.meshgrid(x_line, y_line, z_line, indexing='ij')
        self.xyz = []
        for i in range(self.multilevel):
            level = 2 ** i
            self.xyz.append(np.stack([self.x[::level, ::level, ::level], self.y[::level, ::level, ::level],
                                      self.z[::level, ::level, ::level]]))

    def forward(self, feats, ref_poses, Ks, bbox, img_shape):
        """
        feats: list(tensor([NV C H W])), NV: number of views
        """
        h, w = img_shape[-2:]
        normalizer = torch.tensor([w, h]).type_as(feats[0])  # NOTE
        ref_poses_h = repeat(torch.eye(4), "X Y -> NV X Y", NV=len(ref_poses)).clone().type_as(ref_poses)
        ref_poses_h[:, :3, :4] = ref_poses
        intrinsics_pad = repeat(torch.eye(4), "X Y -> NV X Y", NV=len(Ks)).clone().type_as(Ks)
        intrinsics_pad[:, :3, :3] = Ks
        w2pixel = intrinsics_pad @ ref_poses_h
        NV, _, _, _ = feats[0].shape
        bbox0 = torch.tensor(bbox[0]).type_as(feats[0])[..., None, None, None]

        # import pdb
        # pdb.set_trace()
        volume_mean_var_all = []
        for i in range(len(feats)):
            # ---- step 1: projection -----------------------------------------------
            volume_xyz_temp = torch.tensor(self.xyz[i]).type_as(w2pixel) + bbox0
            volume_xyz = volume_xyz_temp.reshape([3, -1])
            volume_xyz_homo = torch.cat([volume_xyz, torch.ones_like(volume_xyz[0:1])], dim=0)  # [4,XYZ]

            volume_xyz_homo_NV = repeat(volume_xyz_homo, "Num4 XYZ -> NV Num4 XYZ", NV=NV)

            # volume project into views
            volume_xyz_pixel_homo = w2pixel @ volume_xyz_homo_NV  # NV 4 4 @ NV 4 XYZ
            volume_xyz_pixel_homo = volume_xyz_pixel_homo[:, :3]
            mask_valid_depth = volume_xyz_pixel_homo[:, 2] > 0  # NV XYZ
            mask_valid_depth = mask_valid_depth.float()

            volume_xyz_pixel = volume_xyz_pixel_homo / volume_xyz_pixel_homo[:, 2:3]
            volume_xyz_pixel = volume_xyz_pixel[:, :2]
            volume_xyz_pixel = rearrange(volume_xyz_pixel, "NV Dim2 XYZ -> NV XYZ Dim2")
            volume_xyz_pixel = volume_xyz_pixel.unsqueeze(2) / normalizer - 1.

            # projection: project all x * y * z points to NV images and sample features
            # grid sample 2D

            volume_feature, mask = grid_sample_2d(feats[i], volume_xyz_pixel)  # NV C XYZ 1, NV XYZ 1

            volume_feature = volume_feature.squeeze(-1)  # NV C XYZ
            mask = mask.squeeze(-1)  # NV XYZ
            mask = mask * mask_valid_depth

            volume_feature = rearrange(volume_feature, "NV C (NumX NumY NumZ) -> NV NumX NumY NumZ C", NV=NV,
                                       NumX=self.xyz[i].shape[1], NumY=self.xyz[i].shape[2], NumZ=self.xyz[i].shape[3])
            mask = rearrange(mask, "NV (NumX NumY NumZ) -> NV NumX NumY NumZ", NV=NV, NumX=self.xyz[i].shape[1],
                             NumY=self.xyz[i].shape[2], NumZ=self.xyz[i].shape[3])

            weight = mask / (torch.sum(mask, dim=1, keepdim=True) + 1e-8)
            weight = weight.unsqueeze(-1)  # NV X Y Z 1

            # ---- step 3: mean, var ------------------------------------------------
            mean = torch.sum(volume_feature * weight, dim=0, keepdim=True)  # 1 X Y Z C
            var = torch.sum(weight * (volume_feature - mean) ** 2, dim=0, keepdim=True)  # 1 X Y Z C
            # volume_pe = self.linear_pe[i](volume_xyz_temp.permute(3,2,1,0)).unsqueeze(0)
            volume_mean_var = torch.cat([mean, var], dim=-1)  # [1 X Y Z C]
            volume_mean_var = volume_mean_var.permute(0, 4, 3, 2, 1)  # [1,C,Z,Y,X]
            # print(volume_mean_var.shape)
            volume_mean_var_all.append(volume_mean_var)
        # ---- step 4: 3D regularization ----------------------------------------
        volume_mean_var_reg = self.volume_regularization(volume_mean_var_all)
        # print(volume_mean_var_reg.shape)

        return volume_mean_var_reg


name2feature_volume = {
    '3d_unet': FeatureVolume,
    '3d_decoder': MultiScaleFeatureVolume
}
