import logging
import math
import os
import random
import clip
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial, reduce
from itertools import islice, cycle
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat
from collections import namedtuple
from tqdm import tqdm
import numpy as np
from inspect import isfunction
from math import pi
from torch import einsum



def to(t):
    return {'dtype': t.dtype, 'device': t.device}


def exists(val):
    return val is not None


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def max_pos_value(t):
    return torch.finfo(t.dtype).max


def temperature_anneal(t, start=1.0, end=1.0,  anneal_mode='cosine'):
    if anneal_mode == 'cosine':
        annealing_fn = lambda r: np.cos(r * np.pi / 2)
    elif anneal_mode == "linear":
        annealing_fn = lambda r: 1 - r
    elif anneal_mode == "square":
        annealing_fn = lambda r: 1 - r ** 2
    elif anneal_mode == "cubic":
        annealing_fn = lambda r: 1 - r ** 3
    else:
        annealing_fn = lambda r: r

    a = start - end
    b = end
    y = a * annealing_fn(t) + b
    return y


def gamma_func(mode="cosine"):
    if mode == "linear":
        return lambda r: 1 - r
    elif mode == "cosine":
        return lambda r: np.cos(r * np.pi / 2)
    elif mode == "square":
        return lambda r: 1 - r**2
    elif mode == "cubic":
        return lambda r: 1 - r ** 3
    else:
        raise NotImplementedError


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """

    indices_to_remove_top_k = torch.zeros_like(logits).bool()
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove_top_k = logits < torch.topk(logits, top_k)[0][..., -1, None]
        # logits[indices_to_remove] = filter_value

    indices_to_remove_top_p = torch.zeros_like(logits).bool()
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove_top_p = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

    indices_to_remove = indices_to_remove_top_k | indices_to_remove_top_p
    logits[indices_to_remove] = filter_value

    return logits



# helper functions

def exists(val):
    return val is not None

def broadcat(tensors, dim = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]

    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))

    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim = dim)

# rotary embedding helper functions

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

def apply_rotary_emb(freqs, t, start_index = 0):
    freqs = freqs.to(t)
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t_left, t, t_right), dim = -1)

# learned rotation helpers

def apply_learned_rotations(rotations, t, start_index = 0, freq_ranges = None):
    if exists(freq_ranges):
        rotations = einsum('..., f -> ... f', rotations, freq_ranges)
        rotations = rearrange(rotations, '... r f -> ... (r f)')

    rotations = repeat(rotations, '... n -> ... (n r)', r = 2)
    return apply_rotary_emb(rotations, t, start_index = start_index)

# classes

class SphereRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
        learned_freq = False
    ):
        super().__init__()
        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'h':
            freqs =  (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim) / max_freq  * pi
        elif freqs_for == 'w':
            freqs = (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim) / max_freq * 2 * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        self.cache = dict()

        if learned_freq:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer('freqs', freqs)

    def rotate_queries_or_keys(self, t, seq_dim = -2):
        device = t.device
        seq_len = t.shape[seq_dim]
        freqs = self.forward(lambda: torch.arange(seq_len, device = device), cache_key = seq_len)
        return apply_rotary_emb(freqs, t)

    def forward(self, t, cache_key = None):
        if exists(cache_key) and cache_key in self.cache:
            return self.cache[cache_key]

        if isfunction(t):
            t = t()

        freqs = self.freqs

        freqs = torch.einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)

        if exists(cache_key):
            self.cache[cache_key] = freqs

        return freqs


class CLIPSimilarity(nn.Module):
    def __init__(self, model_filename):
        super(CLIPSimilarity, self).__init__()
        self.model_filename = model_filename
        self.model, self.p = clip.load(model_filename, device='cpu')
        self.model = self.model.eval()

    def forward(self, text, image, batch_size=None):
        '''
        text: [X]
          for example,  ['str_1', 'str_2', ..., 'str_X']
        image: [Y, c, w, h]

        return: [X, Y]
        '''
        device = image.device
        img_input = F.interpolate(image, size=224)
        image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
        image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
        img_input = (img_input.clamp(min=-1.0, max=1.0) + 1.0) / 2.0
        img_input -= image_mean[:, None, None]
        img_input /= image_std[:, None, None]
        text_input = clip.tokenize(text, truncate=True).to(device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_input).float()
            image_features = self.model.encode_image(img_input).float()

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            logit_scale = 1.0  # Default value is model.logit_scale.exp()
            if batch_size:
                text_features = rearrange(text_features, '(b x) d -> b x d', b=batch_size)
                image_features = rearrange(image_features, '(b y) d -> b d y', b=batch_size)
            else:
                image_features = image_features.t()
            logits_per_text = logit_scale * torch.matmul(text_features, image_features)
        similarity = logits_per_text
        return similarity


class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        """
        Inputs:
            x: [n,c]
            target: [n]
        """
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class ParameterList(object):
    """a mock parameter list object until below issue is resolved
        https://github.com/pytorch/pytorch/issues/36035
    """

    def __init__(self, kls, prefix, length):
        self.ind = 0
        self.kls = kls
        self.prefix = prefix
        self.length = length

    def _keyname(self, prefix, ind):
        return f'{prefix}_{ind}'

    def append(self, x):
        setattr(self.kls, self._keyname(self.prefix, self.ind), x)
        self.ind += 1

    def to_list(self):
        return [getattr(self.kls, self._keyname(self.prefix, i)) for i in range(self.length)]


class FullSelfAttention(nn.Module):
    def __init__(self, dim, causal=True, heads=8, dim_head=64, one_kv_head=False, attn_dropout=0.1, ff_dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.one_kv_head = one_kv_head
        self.scale = dim_head ** -0.5
        self.causal = causal

        self.q_linear = nn.Linear(dim, inner_dim, bias=False)
        kv_dim = dim_head if one_kv_head else inner_dim
        self.k_linear = nn.Linear(dim, kv_dim, bias=False)
        self.v_linear = nn.Linear(dim, kv_dim, bias=False)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.ff_dropout = nn.Dropout(ff_dropout)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, mask=None):
        """
            x:[b,n,d]
        """
        h, device = self.heads, x.device

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', d=self.dim_head),
                      [self.q_linear(x), self.k_linear(x), self.v_linear(x)])
        if self.one_kv_head:
            k, v = map(lambda t: t.expand(-1, h, -1, -1), (k, v))  # [b,1|h,n,d] -> [b,h,n,d]

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = max_neg_value(dots)
        if exists(mask):
            mask = rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(mask, mask_value)
            del mask

        if self.causal:
            i, j = dots.shape[-2:]
            mask = torch.ones(i, j, device=device).triu_(j - i + 1).bool()
            dots.masked_fill_(mask, mask_value)

        attn = dots.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = self.ff_dropout(out)

        return out


class FullCrossAttention(nn.Module):
    def __init__(self, dim, causal=False, heads=8, dim_head=64, one_kv_head=False, attn_dropout=0.1, ff_dropout=0.1,
                 seq_len=256, mem_expand_size=None, freqs=None):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.one_kv_head = one_kv_head

        self.seq_len = seq_len
        self.mem_expand_size = mem_expand_size
        self.causal = causal
        self.freqs = freqs

        self.q_linear = nn.Linear(dim, inner_dim, bias=False)
        kv_dim = dim_head if one_kv_head else inner_dim
        self.k_linear = nn.Linear(dim, kv_dim, bias=False)
        self.v_linear = nn.Linear(dim, kv_dim, bias=False)

        if exists(mem_expand_size):
            self.rel_pos_h = self.mem_expand_size['h'] * 2 + 1
            self.rel_pos_w = self.mem_expand_size['w'] * 2 + 1
            self.rel_pos_t = self.mem_expand_size['t'] * 2 + 1
            self.rel_pos_frame_size = self.rel_pos_h * self.rel_pos_w

            rpe_heads = 1 if one_kv_head else heads
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.rel_pos_h * self.rel_pos_w * self.rel_pos_t, rpe_heads))
            self.relative_position_bias_table.data.normal_(mean=0, std=0.02)
        else:
            self.relative_position_bias_table = nn.Identity()

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.ff_dropout = nn.Dropout(ff_dropout)
        self.to_out = nn.Linear(inner_dim, dim)

    def get_relative_pos_idxs(self, cur_patch_idx=0, patch_h=4, patch_w=4, mem_idxs=None, locate_fn=None):
        relative_pos_idxs = []
        locate_fn = locate_fn if locate_fn else partial(get_pos_from_idx_down_right)
        locate_fn = partial(locate_fn, patch_h=patch_h, patch_w=patch_w)

        pos_src = locate_fn(idx=cur_patch_idx)
        for tgt_idx in mem_idxs:
            pos_tgt = locate_fn(idx=tgt_idx)

            row_diff = pos_tgt[0] - pos_src[0] + self.mem_expand_size['h']
            col_diff = pos_tgt[1] - pos_src[1] + self.mem_expand_size['w']
            temp_diff = pos_tgt[2] - pos_src[2] + self.mem_expand_size['t']

            row_diff *= self.rel_pos_w
            temp_diff *= self.rel_pos_frame_size

            relative_pos_idxs.append(row_diff + col_diff + temp_diff)
        return relative_pos_idxs
    
    def get_relative_pos_freqs(self, cur_patch_idx=0, patch_h=4, patch_w=4, mem_idxs=None, locate_fn=None, patch_size=16):
        relative_pos_idxs = []
        locate_fn = locate_fn if locate_fn else partial(get_pos_from_idx_down_right)
        locate_fn = partial(locate_fn, patch_h=patch_h, patch_w=patch_w)

        pos_src = locate_fn(idx=cur_patch_idx)
        src_range = [[pos_src[0] * patch_size, (pos_src[0] + 1) * patch_size], [pos_src[1] * patch_size, (pos_src[1] + 1) * patch_size]]
        src_freq = self.freqs[src_range[0][0]:src_range[0][1], src_range[1][0]:src_range[1][1]]
        target_freqs = []
        for tgt_idx in mem_idxs:
            pos_tgt = locate_fn(idx=tgt_idx)
            target_range = [[pos_tgt[0] * patch_size, (pos_tgt[0] + 1) * patch_size], [pos_tgt[1] * patch_size, (pos_tgt[1] + 1) * patch_size]]
            target_freqs.append(self.freqs[target_range[0][0]:target_range[0][1], target_range[1][0]:target_range[1][1]])
        return src_freq, target_freqs

    def forward(self, x, cond=None, mask=None, patch_info=(4, 4), cur_patch_idx=0, locate_fn=None, mem_idxs=None):
        """
            x:[b,n,d]  cond:[b,p*n+text,d]
        """
        b, n, d, h, device = *x.shape, self.heads, x.device
        q = x
        k = v = cond

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', d=self.dim_head),
                      [self.q_linear(q), self.k_linear(k), self.v_linear(v)])

        mem_num = len(mem_idxs)
        if mem_num > 0:
            relative_pos_idxs = self.get_relative_pos_idxs(cur_patch_idx=cur_patch_idx,
                                                           patch_h=patch_info[0], patch_w=patch_info[1],
                                                           mem_idxs=mem_idxs, locate_fn=locate_fn)
            # len:mem
            relative_pos_idxs = torch.LongTensor(relative_pos_idxs).to(device)
            # [mem,h]
            relative_pos_emb = self.relative_position_bias_table[relative_pos_idxs]
            # [mem, h] -> [1, h, mem, 1]
            relative_pos_emb = repeat(relative_pos_emb, 'n h -> b h n d', b=1, d=1)
            # k:[b,p*n+text,d]
            k[:, :, :mem_num * self.seq_len, :] += repeat(relative_pos_emb[:, :, :mem_num, :],
                                                          'b h n d -> b h (n l) d', l=self.seq_len)
            
            src_freq, target_freqs = self.get_relative_pos_freqs(cur_patch_idx=cur_patch_idx,
                                                                    patch_h=patch_info[0], patch_w=patch_info[1],
                                                                    mem_idxs=mem_idxs, locate_fn=locate_fn)
            src_frq_flat = rearrange(src_freq, 'h w d-> (h w) d')
            target_freqs_flat = [rearrange(target_freq, 'h w d-> (h w) d') for target_freq in target_freqs]
            target_freqs_flat = torch.cat(target_freqs_flat, dim=0)
            q = apply_rotary_emb(src_frq_flat, q)
            k[:, :, :mem_num * self.seq_len, :] = apply_rotary_emb(target_freqs_flat, k[:, :, :mem_num * self.seq_len, :])

        if self.one_kv_head:
            k, v = map(lambda t: t.expand(-1, h, -1, -1), (k, v))  # [b,1/h,n,d] -> [b,h,n,d]

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = max_neg_value(dots)
        if exists(mask):
            mask = rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = self.ff_dropout(out)
        return out


class LayerScale(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)

    def forward(self, x):
        return x * self.scale


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, dropout=0., mult=4.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class SingleTransformer(nn.Module):
    def __init__(self, self_attention, cross_attention, ff, dim, depth):
        super().__init__()
        self.self_attention_norm = nn.LayerNorm(dim)
        self.self_attention = self_attention
        self.self_attention_scale = LayerScale(dim, depth)

        if cross_attention is not None:
            self.cross_attention_norm = nn.LayerNorm(dim)
            self.cross_attention = cross_attention
            self.cross_attention_scale = LayerScale(dim, depth)

        self.ff_norm = nn.LayerNorm(dim)
        self.ff = ff
        self.ff_scale = LayerScale(dim, depth)

    def forward(self, x, mask, cond=None, mask_cond=None, patch_info=(4, 4), cur_patch_idx=0,
                locate_fn=None, mem_idxs=None):
        """
            x:[b,n,d]    cond:[b,p*mem+text,d]
        """
        # self-attention
        att = self.self_attention_norm(x)
        att = self.self_attention(x=att, mask=mask)
        att = self.self_attention_scale(att)
        x = x + att

        # cross-attention
        if cond is not None:
            att = self.cross_attention_norm(x)
            att = self.cross_attention(x=att, cond=cond, mask=mask_cond, patch_info=patch_info,
                                       cur_patch_idx=cur_patch_idx, locate_fn=locate_fn, mem_idxs=mem_idxs)
            att = self.cross_attention_scale(att)
            x = x + att

        # feedforward
        ff = self.ff_norm(x)
        ff = self.ff(ff)
        ff = self.ff_scale(ff)
        ff = x + ff
        return ff


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads=8, dim_head=64, one_kv_head=False, ff_mult=4, attn_dropout=0.1, ff_dropout=0.1,
                 seq_len=256,  mem_expand_size=None, causal=True, self_attn_types=('full',),
                 cond=False, cross_attn_types=('full',), use_checkpoint=False, img_full_h = None, img_full_w=None, **kwargs):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.layers = nn.ModuleList([])
        pos_emb_h = SphereRotaryEmbedding(
                    dim = dim_head // 2,
                    freqs_for = 'h',
                    max_freq = img_full_h
        )
        pos_emb_w = SphereRotaryEmbedding(
                    dim = dim_head // 2,
                    freqs_for = 'h',
                    max_freq = img_full_w
        )
        freqs_h = pos_emb_h(torch.linspace(0, img_full_h, steps = img_full_h), cache_key = img_full_h)
        freqs_w = pos_emb_w(torch.linspace(-1, img_full_w, steps = img_full_w), cache_key = img_full_w)
        
        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim = -1)

        self_attn_type_layer = islice(cycle(self_attn_types), depth)
        cross_attn_type_layer = islice(cycle(cross_attn_types), depth)

        for ind, self_attn_type, cross_attn_type in zip(range(depth), self_attn_type_layer, cross_attn_type_layer):
            # self-attention
            if self_attn_type == 'full':
                self_attn_class = FullSelfAttention
            else:
                raise ValueError(f'self attention type "{self_attn_type}" is not valid')

            # cross-attention
            if cross_attn_type == 'full':
                cross_attn_class = FullCrossAttention
            else:
                raise ValueError(f'cross attention type "{self_attn_type}" is not valid')

            self.layers.append(SingleTransformer(
                self_attn_class(causal=causal, dim=dim, heads=heads, dim_head=dim_head, one_kv_head=one_kv_head,
                                attn_dropout=attn_dropout, ff_dropout=ff_dropout),
                cross_attn_class(causal=False, dim=dim, heads=heads, dim_head=dim_head, one_kv_head=one_kv_head,
                                 attn_dropout=attn_dropout, ff_dropout=ff_dropout,
                                 seq_len=seq_len, mem_expand_size=mem_expand_size, freqs=freqs) if cond else None,
                FeedForward(dim, mult=ff_mult, dropout=ff_dropout),
                dim, ind + 1
            ))

    def forward(self, x, mask=None, cond=None, mask_cond=None, patch_info=(4, 4), cur_patch_idx=0,
                locate_fn=None, mem_idxs=None):
        layers_num = len(self.layers)
        for lid in range(layers_num):
            layer = self.layers[lid]
            if self.use_checkpoint:
                x = layer(x, mask, cond, mask_cond, patch_info, cur_patch_idx, locate_fn, mem_idxs)
            else:
                x = layer(x, mask, cond, mask_cond, patch_info, cur_patch_idx, locate_fn, mem_idxs)
        return x



"""
    NUWA-Infinity Wrapper
    
"""

# mems:[b, p*n, d]  idxs:[p]   p:patch_num
Memory = namedtuple('Memory', ['mems', 'idxs'])


# down-right
def get_pos_from_idx_down_right(idx, patch_h, patch_w):
    num_frame = patch_h * patch_w
    idx_inner = idx % num_frame
    cur_temp = idx // num_frame
    cur_row = idx_inner % patch_h
    cur_col = idx_inner // patch_h
    return cur_row, cur_col, cur_temp


def get_idx_from_pos_down_right(cur_row, cur_col, cur_temp, patch_h, patch_w):
    num_frame = patch_h * patch_w
    cur_col = (cur_col + patch_w) % patch_w
    idx = cur_col * patch_h + cur_row + cur_temp * num_frame
    return idx


def get_mem_mapping(img_patch_h, img_patch_w, img_patch_t, mem_expand_size,
                    get_pos_from_idx, get_idx_from_pos):
    """
        Return  mem_mapping, max_used_mapping
    """
    mem_mapping = dict()
    patch_seq_len = img_patch_t * img_patch_h * img_patch_w

    for cur_patch_idx in range(patch_seq_len):
        cur_patch_row, cur_patch_col, cur_patch_temp = get_pos_from_idx(idx=cur_patch_idx)

        rows = torch.arange(max(0, cur_patch_row - mem_expand_size['h']),
                            min(img_patch_h, cur_patch_row + mem_expand_size['h'] + 1))
        cols = torch.arange(max(0, cur_patch_col - mem_expand_size['w']),
                            min(img_patch_w, cur_patch_col + mem_expand_size['w'] + 1))
        temps = torch.arange(max(0, cur_patch_temp - mem_expand_size['t']),
                             min(img_patch_t, cur_patch_temp + mem_expand_size['t'] + 1))
        coords = torch.stack(torch.meshgrid([rows, cols, temps]), dim=-1).reshape(-1, 3)
        mem_idxs = [int(get_idx_from_pos(cur_row=coord[0], cur_col=coord[1], cur_temp=coord[2])) for coord in coords]
        mem_idxs = sorted([idx for idx in mem_idxs if idx < cur_patch_idx])
        mem_mapping[cur_patch_idx] = mem_idxs

    # max_used_mapping {cur_idx: max memory_idx used}
    max_used_mapping = {idx: -1 for idx in range(patch_seq_len)}
    for patch_idx, mem_idxs in mem_mapping.items():
        for idx in mem_idxs:
            max_used_mapping[idx] = patch_idx

    return mem_mapping, max_used_mapping


def get_mem_mapping_pano(img_patch_h, img_patch_w, img_patch_t, mem_expand_size,
                    get_pos_from_idx, get_idx_from_pos):
    """
        Return  mem_mapping, max_used_mapping
    """
    mem_mapping = dict()
    patch_seq_len = img_patch_t * img_patch_h * img_patch_w

    for cur_patch_idx in range(patch_seq_len):
        cur_patch_row, cur_patch_col, cur_patch_temp = get_pos_from_idx(idx=cur_patch_idx)

        rows = torch.arange(max(0, cur_patch_row - mem_expand_size['h']),
                            min(img_patch_h, cur_patch_row + mem_expand_size['h'] + 1))
        cols = torch.arange(cur_patch_col - mem_expand_size['w'],
                            cur_patch_col + mem_expand_size['w'] + 1)
        temps = torch.arange(max(0, cur_patch_temp - mem_expand_size['t']),
                             min(img_patch_t, cur_patch_temp + mem_expand_size['t'] + 1))
        coords = torch.stack(torch.meshgrid([rows, cols, temps]), dim=-1).reshape(-1, 3)
        mem_idxs = [int(get_idx_from_pos(cur_row=coord[0], cur_col=coord[1], cur_temp=coord[2])) for coord in coords]
        # mem_idxs = sorted([idx for idx in mem_idxs if idx < cur_patch_idx])
        mem_mapping[cur_patch_idx] = mem_idxs

    # max_used_mapping {cur_idx: max memory_idx used}
    max_used_mapping = {idx: -1 for idx in range(patch_seq_len)}
    for patch_idx, mem_idxs in mem_mapping.items():
        for idx in mem_idxs:
            max_used_mapping[idx] = patch_idx

    return mem_mapping, max_used_mapping


def update_mem(cur_patch_idx, mem_pool, max_used_mapping, new_mem, clean = False):
    # add new_mem
    mem_pool[cur_patch_idx] = new_mem

    if clean:
        # remove expired memory
        del_mem_idxs = [key for key in mem_pool.keys() if max_used_mapping[key] <= cur_patch_idx]

        for del_idx in del_mem_idxs:
            del mem_pool[del_idx]


def get_mem_from_pool(cur_patch_idx, mem_pool, mem_mapping):
    """
        mem_pool dict: mem_pool[idx]:[b,n,d]
        Return
            mem: Memory(mems=[b,p*n,d], idxs=[p])
    """
    mem_idxs = mem_mapping[cur_patch_idx]
    mem_content = [mem_pool[idx] for idx in mem_idxs]
    mem = Memory(torch.cat(mem_content, dim=-2), mem_idxs) if len(mem_content) != 0 else Memory(None, [])

    return mem


class IterativeModel:
    def __init__(self, args):
        self.args = args
        self.mem_expand_size = args.mem_expand_size
        self.mem_expand_size_pano = args.mem_expand_size_pano

        self.img_w = max(1, self.args.img_size // self.args.compress_ratio)  # 256/16=16
        self.img_h = max(1, self.args.img_size // self.args.compress_ratio)
        self.img_t = 1
        self.image_seq_len = self.img_t * self.img_h * self.img_w  # 1*16*16=256

        img_full_h = self.args.img_full_size_h // self.args.compress_ratio  # 1024/16=64
        img_full_w = self.args.img_full_size_w // self.args.compress_ratio
        img_full_t = self.args.max_video_len

        self.patch_h = img_full_h // self.img_h  # 64//16=4
        self.patch_w = img_full_w // self.img_w
        self.patch_t = img_full_t // self.img_t  # 1//1=1
        self.patch_seq_len = self.patch_t * self.patch_h * self.patch_w  # 1*4*4=16

        get_pos_from_idx = partial(args.get_pos_from_idx, patch_h=self.patch_h, patch_w=self.patch_w)
        get_idx_from_pos = partial(args.get_idx_from_pos, patch_h=self.patch_h, patch_w=self.patch_w)

        self.mem_mapping, self.max_used_mapping = get_mem_mapping(
            img_patch_h=self.patch_h, img_patch_w=self.patch_w,
            img_patch_t=self.patch_t, mem_expand_size=self.mem_expand_size,
            get_pos_from_idx=get_pos_from_idx, get_idx_from_pos=get_idx_from_pos)


        self.mem_mapping_pano, self.max_used_mapping_pano = get_mem_mapping_pano(            
            img_patch_h=self.patch_h, img_patch_w=self.patch_w,
            img_patch_t=self.patch_t, mem_expand_size=self.mem_expand_size_pano,
            get_pos_from_idx=get_pos_from_idx, get_idx_from_pos=get_idx_from_pos)
        
        # memory pool
        self.mem_pool = dict()

    def forward(self, model, inputs):
        image = inputs['label_imgs']

        if hasattr(model, 'module'):
            image_tokens = model.module.get_codebook_indices(image)
        else:
            image_tokens = model.get_codebook_indices(image)

        # down_right
        patches = rearrange(image_tokens, 'b l (ph h) (pw w) -> (l pw ph) b (h w)', ph=self.patch_h, pw=self.patch_w)

        for patch_idx, patch in enumerate(patches):
            # mem : [b,n*p,d], p*[idx]
            memory = get_mem_from_pool(cur_patch_idx=patch_idx, mem_pool=self.mem_pool, mem_mapping=self.mem_mapping)
            inputs['patch_idx'] = patch_idx
            inputs['patch'] = patch
            inputs['memory'] = memory

            # model forward
            outputs = model(inputs)

            # update current memory
            update_mem(cur_patch_idx=patch_idx, mem_pool=self.mem_pool, max_used_mapping=self.max_used_mapping,
                       new_mem=outputs['new_mem'].detach(), clean=False)
            del outputs['new_mem']

            yield outputs
        # print("second_pass")
        for patch_idx, patch in enumerate(patches):
            # mem : [b,n*p,d], p*[idx]
            memory = get_mem_from_pool(cur_patch_idx=patch_idx, mem_pool=self.mem_pool, mem_mapping=self.mem_mapping_pano)
            inputs['patch_idx'] = patch_idx
            inputs['patch'] = patch
            inputs['memory'] = memory
            # print("memory idx", memory.idxs)

            # model forward
            outputs = model(inputs)

            # update current memory
            update_mem(cur_patch_idx=patch_idx, mem_pool=self.mem_pool, max_used_mapping=self.max_used_mapping_pano,
                       new_mem=outputs['new_mem'].detach(), clean=True)
            del outputs['new_mem']

            yield outputs

