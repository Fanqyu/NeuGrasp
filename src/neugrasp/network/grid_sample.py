import torch.nn.functional as F


def grid_sample_2d(input, grid):
    """
    input: L C H W
    grid: L RN SN 2
    """

    N, C, H, W = input.shape
    _, RN, SN, _ = grid.shape
    mask = (grid[..., 0] <= 1.) & \
           (grid[..., 0] >= -1.) & \
           (grid[..., 1] <= 1.) & \
           (grid[..., 1] >= -1.)  # (B L) RN SN
    mask = mask.float()
    output = F.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=True)  # L C RN SN
    return output, mask


def grid_sample_3d(input, grid):
    """
    input: B C X Y Z
    grid: B 1 RN SN 3
    """

    B, C, X, Y, Z = input.shape
    _, _, RN, SN, _ = grid.shape
    output = F.grid_sample(input, grid, mode='bilinear', padding_mode='zeros')  # B C 1 RN SN
    # When mode='bilinear' and the input is 5-D, the interpolation mode used internally will actually be trilinear.
    # However, when the input is 4-D, the interpolation mode will legitimately be bilinear.
    return output.squeeze(2)  # B C RN SN
