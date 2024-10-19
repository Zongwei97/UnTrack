import math
import logging
import pdb
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

from lib.models.layers.patch_embed import PatchEmbed
from .utils import combine_tokens, recover_tokens, token2feature, feature2token
from .vit import VisionTransformer
from ..layers.attn_blocks import CEBlock, candidate_elimination_prompt
import functools
from torch import nn, Tensor

_logger = logging.getLogger(__name__)


class PredictorConv(nn.Module):
    def __init__(self, embed_dim=384, num_modals=2):
        super().__init__()
        self.num_modals = num_modals
        self.score_nets = nn.ModuleList([nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=(embed_dim)),
            nn.Conv2d(embed_dim, 1, 1),
            nn.Sigmoid()
        ) for _ in range(num_modals)])

    def forward(self, x):
        B, C, H, W = x[0].shape

        x_ = [torch.zeros((B, 1, H, W)) for _ in range(self.num_modals)]

        for i in range(self.num_modals):
            x_[i] = self.score_nets[i](x[i])
        return x_


class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


class ConvLayerNorm(nn.Module):
    """Channel first layer norm
    """

    def __init__(self, normalized_shape, eps=1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class LayerNormParallel(nn.Module):
    def __init__(self, num_features, num_modals=4):
        super(LayerNormParallel, self).__init__()
        # self.num_modals = num_modals
        for i in range(num_modals):
            setattr(self, 'ln_' + str(i), ConvLayerNorm(num_features, eps=1e-6))

    def forward(self, x_parallel):
        return [getattr(self, 'ln_' + str(i))(x) for i, x in enumerate(x_parallel)]


class PatchEmbedParallel(nn.Module):
    def __init__(self, c1=3, c2=32, patch_size=7, stride=4, padding=0, num_modals=4):
        super().__init__()
        self.proj = ModuleParallel(nn.Conv2d(c1, c2, 3, 1, 1))  # padding=(ps[0]//2, ps[1]//2)
        self.norm = LayerNormParallel(c2, num_modals)

    def forward(self, x: list) -> list:
        x = self.proj(x)
        _, _, H, W = x[0].shape
        x = self.norm(x)
        return x, H, W


class Fovea(nn.Module):

    def __init__(self, smooth=False):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)

        self.smooth = smooth
        if smooth:
            self.smooth = nn.Parameter(torch.zeros(1) + 10.0)

    def forward(self, x):
        '''
            x: [batch_size, features, k]
        '''
        b, c, h, w = x.shape
        x = x.contiguous().view(b, c, h * w)

        if self.smooth:
            mask = self.softmax(x * self.smooth)
        else:
            mask = self.softmax(x)
        output = mask * x
        output = output.contiguous().view(b, c, h, w)

        return output


class _LoRALayer(nn.Module):
    def __init__(self, w: nn.Module, w_a: nn.Module, w_b: nn.Module):
        super().__init__()
        self.lora_attn = w
        self.prompt_w_a = w_a
        self.prompt_w_b = w_b

    def forward(self, x, mask_x=None, **kwargs):
        x_attn, attn = self.lora_attn(x, mask_x, True)
        x = x_attn + self.prompt_w_b(self.prompt_w_a(x))
        return x, attn


class PredictorLG(nn.Module):
    """ Image to Patch Embedding from DydamicVit
    """

    def __init__(self, embed_dim=384):
        super().__init__()
        self.num_parallel = 2
        self.score_nets = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1)
        )

    def forward(self, x):
        x = self.score_nets(x)
        return x


class TokenExchange(nn.Module):
    def __init__(self):
        super(TokenExchange, self).__init__()

    def forward(self, x, mask, mask_threshold):
        # x: [B, N, C], mask: [B, N, 1]
        x0, x1 = torch.zeros_like(x[0]), torch.zeros_like(x[1])
        x1[mask[1] >= mask_threshold] = x[1][mask[1] >= mask_threshold]
        x1[mask[1] < mask_threshold] = x[0][mask[1] < mask_threshold]
        return x1


class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


class LayerNormParallel(nn.Module):
    def __init__(self, num_features):
        super(LayerNormParallel, self).__init__()
        self.num_parallel = 2
        for i in range(self.num_parallel):
            setattr(self, 'ln_' + str(i), nn.LayerNorm(num_features, eps=1e-6))

    def forward(self, x_parallel):
        return [getattr(self, 'ln_' + str(i))(x) for i, x in enumerate(x_parallel)]


class SelectDillDeepFuse(nn.Module):
    def  __init__(self, embed_dim):
        super(SelectDillDeepFuse, self).__init__()
        self.gumbel_x = nn.Linear(embed_dim, 1)
        self.gumbel_mod = nn.Linear(embed_dim, 1)
        rank = 8
        self.proj_x = nn.Linear(embed_dim, rank)
        self.proj_mod = nn.Linear(embed_dim, rank)
        self.proj_res_x = nn.Linear(embed_dim, rank)
        self.proj_res_mod = nn.Linear(embed_dim, rank)

        self.proj_fuse_lvl_x = nn.Linear(2*rank, rank)
        self.proj_fuse_lvl_mod = nn.Linear(2*rank, rank)
        self.proj_back = nn.Linear(rank, embed_dim, True)

    def forward(self, x):
        x, mod = x.chunk(2,dim=1)
        int = x
        x = feature2token(x)
        mod = feature2token(mod)
        B, N, emb_dim = x.shape[0], x.shape[1], x.shape[2]
        number = N//4
        tau = 1

        token_scores = self.gumbel_x(x) #[64 256 1]
        token_scores = token_scores.reshape(B, -1) #[64 256]
        token_mask, rej_mask = gumbel_softmax(F.log_softmax(token_scores, dim=-1), k=number, tau=tau, hard=True) #[64 256]  sum[token==1] = 64*number, from each batch select the n most tokens
        #token_mask[:, 0] = 1.
        token_mask_x = token_mask.expand(emb_dim, -1, -1).permute(1, 2, 0)
        rej_mask_x = rej_mask.expand(emb_dim, -1, -1).permute(1, 2, 0)
        low_x = self.proj_x(x * token_mask_x + mod * rej_mask_x)

        low_res_x = self.proj_res_x((x + mod) * (1 - rej_mask_x - token_mask_x))

        fuse_x = self.proj_fuse_lvl_x(torch.cat((low_x, low_res_x), dim=2) )

        token_scores = self.gumbel_mod(mod) #[64 256 1]
        token_scores = token_scores.reshape(B, -1) #[64 256]
        token_mask, rej_mask = gumbel_softmax(F.log_softmax(token_scores, dim=-1), k=number, tau=tau, hard=True) #[64 256]  sum[token==1] = 64*number, from each batch select the n most tokens
        #token_mask[:, 0] = 1.
        token_mask_mod = token_mask.expand(emb_dim, -1, -1).permute(1, 2, 0)
        rej_mask_mod = rej_mask.expand(emb_dim, -1, -1).permute(1, 2, 0)

        low_mod = self.proj_mod(mod * token_mask_mod + x * rej_mask_mod)
        low_res_mod = self.proj_res_mod((x + mod) * (1 - rej_mask_mod - token_mask_mod))

        fuse_mod = self.proj_fuse_lvl_mod(torch.cat((low_mod, low_res_mod), dim=2) )

        fuse = self.proj_back(fuse_x + fuse_mod)
        return token2feature(fuse)



class EdgeLora(nn.Module):
    def  __init__(self, embed_dim):
        super(EdgeLora, self).__init__()
        rank = 4
        self.proj_d = nn.Linear(embed_dim, rank)
        self.proj_t = nn.Linear(embed_dim, rank)
        self.proj_e = nn.Linear(embed_dim, rank)

        self.proj_grad = nn.Linear(embed_dim, rank)

        self.proj_fuse = nn.Linear(3*rank, rank)

        self.fuse = nn.Linear(rank, rank)
        self.proj_back = nn.Linear(rank, embed_dim, True)

    def forward(self, x, grad, sem_idx=None):
        d = torch.zeros_like(x)
        t = torch.zeros_like(x)
        e = torch.zeros_like(x)

        d[sem_idx == 1, ...] = x[sem_idx == 1, ...]
        t[sem_idx == 2, ...] = x[sem_idx == 2, ...]
        e[sem_idx == 3, ...] = x[sem_idx == 3, ...]

        rank_d = self.proj_d(d)
        rank_t = self.proj_t(t)
        rank_e = self.proj_e(e)
        rank_grad = self.proj_grad(grad)
        fused = self.proj_fuse(torch.cat((rank_d , rank_t , rank_e), dim=2))
        guide = self.fuse(rank_grad)
        rank =  fused + guide
        recon = self.proj_back(rank)
        return recon + grad



def gradient(depth_tmp, step =7):
    B, C, H, W = depth_tmp.size()
    pad = (step - 1) // 2
    depth_tmp = F.pad(depth_tmp, [pad, pad, pad, pad], mode='constant', value=0)
    patches = depth_tmp.unfold(dimension=2, size=step, step=1)
    patches = patches.unfold(dimension=3, size=step, step=1)
    max_depth, _ = patches.reshape(B, C, H, W, -1).max(dim=-1)

    step = float(step)
    shift_list = [[step / H, 0.0 / W], [-step / H, 0.0 / W], [0.0 / H, step / W], [0.0 / H, -step / W]]
    output_list = []
    for shift in shift_list:
        transform_matrix = torch.tensor([[1, 0, shift[0]], [0, 1, shift[1]]]).unsqueeze(0).repeat(B, 1, 1).to(depth_tmp.device)
        grid = F.affine_grid(transform_matrix, max_depth.shape).float()
        output = F.grid_sample(max_depth, grid, mode='nearest')
        output = max_depth - output
        output_mask = ((output == max_depth) == False)
        output = output * output_mask
        output_list.append(output)
    grad = torch.cat(output_list, dim=1)
    max_grad = torch.abs(grad).max(dim=1)[0].unsqueeze(1)

    return grad, max_grad

def scatter(logits, index, k):
    bs = logits.shape[0]
   #print('bs = {}'.format(bs))

    x_index = torch.arange(bs).reshape(-1, 1).expand(bs,k)
    x_index = x_index.reshape(-1).tolist()
    y_index = index.reshape(-1).tolist()

    output = torch.zeros_like(logits).cuda()
    output[x_index, y_index] = 1.0
   #print(output.sum(dim=1))

    return output

def gumbel_softmax(logits, k, tau=1, hard=False, eps=1e-10, dim=-1):
    # type: (torch.Tensor, float, bool, float, int) -> torch.Tensor
    #gumbels = (
    #    -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    #)  # ~Gumbel(0,1)
    #gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    gumbels = logits
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.topk(k, dim=dim)[1]
        y_hard = scatter(logits, index, k)
        ret_top = y_hard - y_soft.detach() + y_soft

        index = (-y_soft).topk(k, dim=dim)[1]
        y_hard = scatter(logits, index, k)
        ret_bot = y_hard - y_soft.detach() + y_soft


    else:
        # Reparametrization trick.
        ret_top = y_soft
        ret_bot = y_soft

    if torch.isnan(ret_top).sum():
        raise OverflowError(f'gumbel softmax output: {ret_top}')
    return ret_top, ret_bot


class VisionTransformerCE(VisionTransformer):
    """ Vision Transformer with candidate elimination (CE) module

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', ce_loc=None, ce_keep_ratio=None, search_size=None, template_size=None,
                 new_patch_size=None, prompt_type=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
            new_patch_size: backbone stride
        """
        super().__init__()

        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.shallow_prompt = EdgeLora(embed_dim)

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.patch_embed_prompt_dte = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.patch_embed_prompt_grad = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans*9, embed_dim=embed_dim)

        self.patch_embed_prompt_edge = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=1, embed_dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_drop = nn.Dropout(p=drop_rate)

        '''
        prompt parameters
        '''
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_search = new_P_H * new_P_W
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_template = new_P_H * new_P_W
        """add here, no need use backbone.finetune_track """
        self.pos_embed_z = nn.Parameter(torch.zeros(1, self.num_patches_template, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.num_patches_search, embed_dim))

        self.prompt_type = prompt_type
        # various architecture
        if self.prompt_type in ['shaw', 'deep']:
            prompt_blocks = []
            block_nums = depth if self.prompt_type == 'deep' else 1
            for i in range(block_nums):
                prompt_blocks.append(SelectDillDeepFuse(embed_dim))

            self.prompt_blocks = nn.Sequential(*prompt_blocks)
            prompt_norms = []
            for i in range(block_nums):
                prompt_norms.append(norm_layer(embed_dim))
            self.prompt_norms = nn.Sequential(*prompt_norms)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        ce_index = 0
        self.ce_loc = ce_loc
        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1

            blocks.append(
                CEBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                    keep_ratio_search=ce_keep_ratio_i)
            )

        self.blocks = nn.Sequential(*blocks)

        self.w_As = []  # These are linear layers
        self.w_Bs = []

        rank = 4
        # Here, we do the surgery
        for t_layer_idx, blk in enumerate(self.blocks):
            # If we only want few lora layer instead of all]
            w_a_linear_qkv = nn.Linear(embed_dim, rank, bias=False)
            w_b_linear_qkv = nn.Linear(rank, embed_dim, bias=False)
            self.w_As.append(w_a_linear_qkv)
            self.w_Bs.append(w_b_linear_qkv)
            # blk.prev_attn = blk.attn
            blk.attn = _LoRALayer(blk.attn, w_a_linear_qkv, w_b_linear_qkv)
        self.reset_parameters()

        self.prompt_model = self.blocks

        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)

    def reset_parameters(self):
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def tokenselect(self, x_ext, module):
        x_scores = module(x_ext)
        for i in range(len(x_ext)):
            x_ext[i] = x_scores[i] * x_ext[i] + x_ext[i]
        x_f = functools.reduce(torch.max, x_ext)
        return x_f

    def forward_features(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False):
        # x = 32, 9, 256, 256; z = 32, 9, 128, 128
        loss_mod = []

        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        sem_idx = x[:, 6, 0, 0]
        # rgb_img

        #import pdb; pdb.set_trace()


        x_rgb = x[:, :3, :, :]
        z_rgb = z[:, :3, :, :]

        x_dte = x[:, 3:6, ...]
        z_dte = z[:, 3:6, ...]

        x, z = x_rgb, z_rgb
        z = self.patch_embed(z)  # 32, 64, 768
        x = self.patch_embed(x)  # 32, 256, 768

        z_rgb_4edge, max_rgb_z_edge = gradient(z_dte)
        x_rgb_4edge, max_rgb_x_edge = gradient(x_dte)

        z_4edge, max_z_edge = gradient(z_dte)
        x_4edge, max_x_edge = gradient(x_dte)


        z_dte_mod = self.patch_embed_prompt_grad(torch.cat((z_rgb, z_rgb_4edge, z_4edge), dim=1))
        x_dte_mod = self.patch_embed_prompt_grad(torch.cat((x_rgb, x_rgb_4edge, x_4edge), dim=1))

        z_dte = self.patch_embed_prompt_dte(z_dte)  # 32, 64, 768
        x_dte = self.patch_embed_prompt_dte(x_dte)

        z_dte = self.shallow_prompt(z_dte, z_dte_mod, sem_idx)
        x_dte = self.shallow_prompt(x_dte, x_dte_mod, sem_idx)

        ze = self.patch_embed_prompt_edge(max_z_edge)
        xe = self.patch_embed_prompt_edge(max_x_edge)


        if self.prompt_type in ['shaw', 'deep']:
            z_feat = token2feature(self.prompt_norms[0](z))
            x_feat = token2feature(self.prompt_norms[0](x))

            z_dte_feat = token2feature(self.prompt_norms[0](z_dte))
            x_dte_feat = token2feature(self.prompt_norms[0](x_dte))

            z_feat = torch.cat([z_feat, z_dte_feat], dim=1)
            x_feat = torch.cat([x_feat, x_dte_feat], dim=1)
            z_feat = self.prompt_blocks[0](z_feat) 
            x_feat = self.prompt_blocks[0](x_feat) 

            z_dte = feature2token(z_feat)
            x_dte = feature2token(x_feat)
            z_prompted, x_prompted = z_dte, x_dte

            z = z + z_dte
            x = x + x_dte
        else:
            z = z + z_dte
            x = x + x_dte

        # attention mask handling
        # B, H, W
        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z += self.pos_embed_z
        x += self.pos_embed_x

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        x = combine_tokens(z, x, mode=self.cat_mode)
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        global_index_t = torch.linspace(0, lens_z - 1, lens_z, dtype=torch.int64).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)

        global_index_s = torch.linspace(0, lens_x - 1, lens_x, dtype=torch.int64).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)

        removed_indexes_s = []
        removed_flag = False
        for i, blk in enumerate(self.prompt_model):
            '''
            add parameters prompt from 1th layer
            '''
            if i >= 1:
                if self.prompt_type in ['deep']:
                    x_ori = x
                    # recover x to go through prompt blocks
                    lens_z_new = global_index_t.shape[1]
                    lens_x_new = global_index_s.shape[1]
                    z = x[:, :lens_z_new]
                    x = x[:, lens_z_new:]
                    if removed_indexes_s and removed_indexes_s[0] is not None:
                        removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)
                        pruned_lens_x = lens_x - lens_x_new
                        pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
                        x = torch.cat([x, pad_x], dim=1)
                        index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
                        C = x.shape[-1]
                        x = torch.zeros_like(x).scatter_(dim=1,
                                                         index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64),
                                                         src=x)
                    x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)
                    x = torch.cat([z, x], dim=1)

                    # prompt
                    x = self.prompt_norms[i - 1](x)
                    z_tokens = x[:, :lens_z, :]
                    x_tokens = x[:, lens_z:, :]

                    z_feat = token2feature(z_tokens)
                    x_feat = token2feature(x_tokens)
                    #import pdb; pdb.set_trace()

                    z_prompted = self.prompt_norms[i](z_prompted) + ze
                    x_prompted = self.prompt_norms[i](x_prompted) + xe

                    z_prompt_feat = token2feature(z_prompted)
                    x_prompt_feat = token2feature(x_prompted)

                    z_feat = torch.cat([z_feat, z_prompt_feat], dim=1)
                    x_feat = torch.cat([x_feat, x_prompt_feat], dim=1)

                    z_feat = self.prompt_blocks[i](z_feat)
                    x_feat = self.prompt_blocks[i](x_feat)

                    z = feature2token(z_feat)
                    x = feature2token(x_feat)
                    z_prompted, x_prompted = z, x

                    x = combine_tokens(z, x, mode=self.cat_mode)
                    # re-conduct CE
                    x = candidate_elimination_prompt(x, global_index_t.shape[1], global_index_s)
                    x = x_ori + x

            x, global_index_t, global_index_s, removed_index_s, attn = \
                blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate)



            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)

        x = self.norm(x)
        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]

        z = x[:, :lens_z_new]
        x = x[:, lens_z_new:]

        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            # recover original token order
            C = x.shape[-1]
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64),
                                             src=x)

        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)

        # re-concatenate with the template, which may be further used by other modules
        x = torch.cat([z, x], dim=1)

        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,  # used for visualization
        }

        return x, aux_dict

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False):

        x, aux_dict = self.forward_features(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate, )

        return x, aux_dict


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            print('Load pretrained OSTrack from: ' + pretrained)
            print(f"missing_keys: {missing_keys}")
            print(f"unexpected_keys: {unexpected_keys}")

    return model


def vit_base_patch16_224_ce_prompt(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224_ce_prompt(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
