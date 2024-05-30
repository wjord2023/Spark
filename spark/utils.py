import torch
from torch import nn
from torch.nn import functional as F


class GlobalSparseAttention(nn.Module):
    def __init__(self, image_size, channel_size, patch_size, num_heads):
        super(GlobalSparseAttention, self).__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.fold = nn.Fold(
            output_size=(image_size, image_size),
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.across_patches_attn = nn.MultiheadAttention(
            embed_dim=image_size**2 // patch_size**2, num_heads=num_heads
        )
        self.in_patch_attn = nn.MultiheadAttention(
            embed_dim=channel_size * patch_size**2, num_heads=num_heads
        )

    def forward(self, x):
        sparse_patches = self.unfold(x)
        sparse_patches = sparse_patches.permute(1, 0, 2)

        global_attn, _ = self.across_patches_attn(
            sparse_patches, sparse_patches, sparse_patches
        )
        global_attn = global_attn.permute(1, 0, 2)

        global_attn_image = self.fold(global_attn)
        patches = self.unfold(global_attn_image)
        patches = patches.permute(2, 0, 1)

        in_patch_attn, _ = self.in_patch_attn(patches, patches, patches)
        in_patch_attn = in_patch_attn.permute(1, 2, 0)

        attn_image = self.fold(in_patch_attn)
        return attn_image + x


class SinusoidalPositionalEncoding2D(nn.Module):
    def __init__(self, height, width):
        super(SinusoidalPositionalEncoding2D, self).__init__()
        self.height = height
        self.width = width

    def forward(self, x):
        pos_encoding = torch.zeros(x.size(0), 2, self.height, self.width, device=x.device)
        pos_h = torch.arange(self.height).unsqueeze(1).expand(-1, self.width)
        pos_w = torch.arange(self.width).unsqueeze(0).expand(self.height, -1)
        pos_encoding[:, 0, :, :] = torch.sin(
            pos_h / (10000 ** (2 * pos_h / self.height))
        )
        pos_encoding[:, 1, :, :] = torch.cos(
            pos_w / (10000 ** (2 * pos_w / self.width))
        )

        return torch.cat([x, pos_encoding], dim=1)
