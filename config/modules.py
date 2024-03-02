import clip
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
from functools import partial, reduce
from itertools import islice, cycle
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat

# TODO this version    use bos +1  && edge of mask attention nearby


def exists(val):
    return val is not None


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def max_pos_value(t):
    return torch.finfo(t.dtype).max


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


class AxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, axial_shape, axial_dims=None):
        super().__init__()

        self.dim = dim
        axial_shape = tuple([shape for shape in axial_shape if shape != 1])  # remove the axis whose shape is 1
        self.shape = axial_shape
        self.max_seq_len = reduce(lambda x, y: x * y, axial_shape, 1)

        self.summed = axial_dims is None
        axial_dims = ((dim,) * len(axial_shape)) if self.summed else axial_dims

        assert len(self.shape) == len(
            axial_dims), 'number of axial dimensions must equal the number of dimensions in the shape'
        assert self.summed or not self.summed and sum(
            axial_dims) == dim, f'axial dimensions must sum up to the target dimension {dim}'

        self.weights = ParameterList(self, 'weights', len(axial_shape))

        for ind, (shape, axial_dim) in enumerate(zip(self.shape, axial_dims)):
            ax_shape = [1] * len(self.shape)
            ax_shape[ind] = shape
            ax_shape = (1, *ax_shape, axial_dim)
            ax_emb = nn.Parameter(torch.zeros(ax_shape).normal_(0, 1))
            self.weights.append(ax_emb)

    def forward(self, x, idx=None, row_range=None, col_range=None):
        b, t, e = x.shape
        assert (t <= self.max_seq_len), f'Sequence length ({t}) must ' \
                                        f'be less than the maximum sequence length allowed ({self.max_seq_len})'
        embs = []

        for ax_emb in self.weights.to_list():
            axial_dim = ax_emb.shape[-1]
            expand_shape = (b, *self.shape, axial_dim)
            emb = ax_emb.expand(expand_shape).reshape(b, self.max_seq_len, axial_dim)
            embs.append(emb)
        pos_emb = sum(embs) if self.summed else torch.cat(embs, dim=-1)

        if idx is None and row_range is None:
            return pos_emb[:, :t].to(x)
        elif idx is not None:
            return pos_emb[:, idx:idx + 1].to(x)
        else:
            rows = torch.arange(row_range[0], row_range[1])
            cols = torch.arange(col_range[0], col_range[1])
            coords = torch.stack(torch.meshgrid([rows, cols]), dim=-1).reshape(-1, 2)
            coords[:, 0] = coords[:, 0] * self.shape[0]
            pos_idxs = coords.sum(dim=-1)
            return pos_emb[:, pos_idxs].to(x)


class PanoAxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, axial_shape, axial_dims=None):
        super().__init__()

        self.dim = dim
        axial_shape = tuple([shape for shape in axial_shape if shape != 1])  # remove the axis whose shape is 1
        self.shape = axial_shape
        self.max_seq_len = reduce(lambda x, y: x * y, axial_shape, 1)

        self.summed = axial_dims is None
        axial_dims = ((dim,) * len(axial_shape)) if self.summed else axial_dims

        assert len(self.shape) == len(
            axial_dims), 'number of axial dimensions must equal the number of dimensions in the shape'
        assert self.summed or not self.summed and sum(
            axial_dims) == dim, f'axial dimensions must sum up to the target dimension {dim}'

        self.weights = ParameterList(self, 'weights', len(axial_shape))

        for ind, (shape, axial_dim) in enumerate(zip(self.shape, axial_dims)):
            ax_shape = [1] * len(self.shape)
            ax_shape[ind] = shape
            ax_shape = (1, *ax_shape, axial_dim)
            ax_emb = nn.Parameter(torch.zeros(ax_shape).normal_(0, 1))
            self.weights.append(ax_emb)

    def forward(self, x, idx=None, row_range=None, col_range=None):
        b, t, e = x.shape
        assert (t <= self.max_seq_len), f'Sequence length ({t}) must ' \
                                        f'be less than the maximum sequence length allowed ({self.max_seq_len})'
        embs = []

        for ax_emb in self.weights.to_list():
            axial_dim = ax_emb.shape[-1]
            expand_shape = (b, *self.shape, axial_dim)
            emb = ax_emb.expand(expand_shape).reshape(b, self.max_seq_len, axial_dim)
            embs.append(emb)
        pos_emb = sum(embs) if self.summed else torch.cat(embs, dim=-1)

        if idx is None and row_range is None:
            return pos_emb[:, :t].to(x)
        elif idx is not None:
            return pos_emb[:, idx:idx + 1].to(x)
        else:
            rows = torch.arange(row_range[0], row_range[1])
            cols = torch.arange(col_range[0], col_range[1])
            coords = torch.stack(torch.meshgrid([rows, cols]), dim=-1).reshape(-1, 2)
            coords[:, 0] = coords[:, 0] * self.shape[0]
            pos_idxs = coords.sum(dim=-1)
            return pos_emb[:, pos_idxs].to(x)



class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super(FixedPositionalEmbedding, self).__init__()
        self.dim = dim
        inv_freq = 1 / (10000 ** (torch.arange(0.0, dim, 2.0) / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[None, :, :].expand(bsz, -1, -1)
        else:
            return pos_emb[None, :, :]


class ParallelAttention(nn.Module):
    """
        moved and modified from DALLE_pytorch
    """

    def __init__(self, dim, causal=True, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.causal = causal

        self.q_linear = nn.Linear(dim, inner_dim, bias=False)
        self.k_linear = nn.Linear(dim, inner_dim, bias=False)
        self.v_linear = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        b, n, _, h, device = *q.shape, self.heads, q.device

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
                      [self.q_linear(q), self.k_linear(k), self.v_linear(v)])

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
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = self.dropout(out)
        return out

    def forward_with_cache(self, q, k, v, cur_idx, mask=None, is_cond=False, layer=0, cache=None):
        b, n, _, h, device = *q.shape, self.heads, q.device

        q = rearrange(self.q_linear(q), 'b n (h d) -> b h n d', h=h)  # [b,1,d]

        layer_name = 'layer%s' % layer
        if not is_cond:
            k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
                       [self.k_linear(k), self.v_linear(v)])  # [b,1,d]
            if cache[layer_name]['k'] is not None:
                k = torch.cat((cache[layer_name]['k'], k), dim=-2)  # cur_idx=0,1,2  cat([b,2,d], [b,1,d]) = [b,3,d]
                v = torch.cat((cache[layer_name]['v'], v), dim=-2)
            cache[layer_name]['k'] = k
            cache[layer_name]['v'] = v
        else:
            if cache[layer_name]['k_cond'] is not None:
                k = cache[layer_name]['k_cond']
                v = cache[layer_name]['v_cond']
            else:
                k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
                           [self.k_linear(k), self.v_linear(v)])  # cross-att  k,v:[b,77,d]->[b,77,d]
                cache[layer_name]['k_cond'] = k
                cache[layer_name]['v_cond'] = v

        # [b,h,37,37]  [b,h,1,38]
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # q[b,1,d]  k[b,3,d] -> [b,1,3]
        mask_value = max_neg_value(dots)

        if exists(mask):
            mask = rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)  # [b,1,3]  v[b,3,d] -> [b,1,d]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = self.dropout(out)  # [b,1,d]
        return out

class AxialAttention(nn.Module):
    """
        moved and modified from DALLE_pytorch
    """

    def __init__(self, dim=512, type=0, causal=True, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.type = type   # 0:row  1:col
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.causal = causal

        self.q_linear = nn.Linear(dim, inner_dim, bias=False)
        self.k_linear = nn.Linear(dim, inner_dim, bias=False)
        self.v_linear = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        b, n, _, hd, device = *q.shape, self.heads, q.device
        h = w = int(np.sqrt(n-1))  # 1025-1 1024 32

        if self.type == 0:
            n_q = h
            split_fn = lambda x: rearrange(x, 'b (h w) (hd d) -> b hd h w d', h=h, hd=hd)
            merge_fn = lambda x: rearrange(x, 'b hd h w d -> b (h w) (hd d)')
        else:
            n_q = w
            split_fn = lambda x: rearrange(x, 'b (h w) (hd d) -> b hd w h d', h=h, hd=hd)
            merge_fn = lambda x: rearrange(x, 'b hd w h d -> b (h w) (hd d)')

        q, k, v = self.q_linear(q), self.k_linear(k), self.v_linear(v)
        q_bos, k_bos, v_bos = map(lambda x: rearrange(x, 'b n (hd d) -> b hd n d', hd=hd),
                                  [q[:, :1], k[:, :1], v[:, :1]])
        q_bos, k_bos, v_bos = map(lambda x: repeat(x, 'b hd n d -> b hd nq n d', nq=n_q),
                                  [q_bos, k_bos, v_bos])

        q_img, k_img, v_img = map(split_fn, [q[:, 1:], k[:, 1:], v[:, 1:]])
        # [b hd 32 1+32 d]
        q, k, v = map(lambda x: torch.cat([x[0], x[1]], dim=-2),
                      [(q_bos, q_img), (k_bos, k_img), (v_bos, v_img)])

        # [b hd 32 33 d] [b hd 32 33 d] -> [b hd 32 33 33]
        dots = torch.einsum('b h n i d, b h n j d -> b h n i j', q, k)* self.scale

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
        # [b hd 32 33 33] [b hd 32 33 d] -> [b hd 32 33 d]
        out = torch.einsum('b h n i j, b h n j d -> b h n i d', attn, v)

        out_bos = out[:, :, 0, :1, :]  # [b hd 1 d]
        out_img = out[:, :, :, 1:, :]  # [b hd 32 32 d]

        out_bos = rearrange(out_bos, 'b hd n d -> b n (hd d)')
        out_img = merge_fn(out_img)

        out = torch.cat([out_bos, out_img], dim=-2)

        out = self.to_out(out)
        out = self.dropout(out)
        return out

    def forward_with_cache(self, q, k, v, cur_idx, mask=None, is_cond=False, layer=0, cache=None):

        return q



def generate_nearby_mask(temporal_size=10, spatial_size=(16, 16), kernel_spatial=5, kernel_temporal=7, causal=False):
    """
        Generate nearby_mask
    """
    assert kernel_spatial % 2 != 0 and kernel_temporal % 2 != 0, 'kernel_size must be odd!'
    temporal, height, width = temporal_size, spatial_size[0], spatial_size[1]
    temporal_extent = (kernel_temporal - 1) // 2
    frame_len = height * width
    seq_len = temporal * frame_len
    padding = kernel_spatial // 2
    if causal:
        mask_template = torch.arange(kernel_spatial ** 2)
        conv_mid = (kernel_spatial ** 2) // 2
        mask_template = (mask_template >= conv_mid).reshape(kernel_spatial, kernel_spatial)
    mask_block = torch.ones(seq_len, temporal, height + padding * 2, width + padding * 2).bool()

    for h in range(height):
        for w in range(width):
            for t in range(temporal):
                if causal:
                    mask_block[t * frame_len + h * width + w][t, h:h + kernel_spatial,
                    w:w + kernel_spatial] = mask_template
                    mask_block[t * frame_len + h * width + w][
                    max(0, t - temporal_extent):t, h:h + kernel_spatial, w:w + kernel_spatial] = False
                else:
                    mask_block[t * frame_len + h * width + w][
                    max(0, t - temporal_extent):min(temporal + 1, t + temporal_extent + 1), h:h + kernel_spatial,
                    w:w + kernel_spatial] = False

    mask_block = mask_block[:, :, padding:-padding, padding:-padding].reshape(seq_len, seq_len)  # [2560,2560]

    # Adapt to the bos pattern
    if causal:
        mask_block = torch.roll(mask_block, shifts=-1, dims=0)
        mask_block = F.pad(mask_block, (1, 0, 1, 0), value=True)  # [2560,2560] -> [1+2560, 1+2560]
        mask_block[0, 0] = False
    else:
        mask_block = F.pad(mask_block, (0, 0, 0, 1), value=True)  # [2560,2560] -> [2560+1, 2560]
    return mask_block


class NearbyAttention(nn.Module):
    def __init__(self, dim, image_size=(16, 16), vid_len=10, heads=8, dim_head=64,
                 dropout=0., sparse_block_info=None, **kwargs):
        super().__init__()
        self.kernel_spatial, self.kernel_temporal, self.causal = sparse_block_info['kernel_spatial'], sparse_block_info[
            'kernel_temporal'], sparse_block_info['causal']

        inner_dim = dim_head * heads
        self.vid_len = vid_len
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.image_size_h, self.image_size_w = image_size

        self.q_linear = nn.Linear(dim, inner_dim, bias=False)
        self.k_linear = nn.Linear(dim, inner_dim, bias=False)
        self.v_linear = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Linear(inner_dim, dim)

        mask_block = generate_nearby_mask(temporal_size=vid_len, spatial_size=image_size,
                                          kernel_spatial=self.kernel_spatial, kernel_temporal=self.kernel_temporal,
                                          causal=self.causal)
        self.register_buffer('mask_block', mask_block)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        b, n, _, h, device = *q.shape, self.heads, q.device

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
                      [self.q_linear(q), self.k_linear(k), self.v_linear(v)])

        dots = torch.einsum('b h i d, b h j d -> b h i j', q,
                            k) * self.scale  # q[b,2560,d] k[b,2560,d] -> [b,2560,2560]  [b,2560,125]

        mask_value = max_neg_value(dots)
        if exists(mask):
            mask = rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(mask, mask_value)
            del mask

        mask = self.mask_block  # [1+2560, 1+2560/2560]

        dots.masked_fill_(mask, mask_value)
        attn = dots.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)  # dot:[b,2560,d]  v[b,2560,d] -> [b,2560,d]
        out = rearrange(out, 'b h n d -> b n (h d)')

        # out = rearrange(q+k+v, 'b h n d -> b n (h d)')

        out = self.to_out(out)
        out = self.dropout(out)
        return out[:, :n]

    def forward_with_cache(self, q, k, v, cur_idx, mask=None, is_cond=False, layer=0, cache=None):
        b, n, _, h, device = *q.shape, self.heads, q.device

        q = rearrange(self.q_linear(q), 'b n (h d) -> b h n d', h=h)

        layer_name = 'layer%s' % layer
        if not is_cond:
            k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
                       [self.k_linear(k), self.v_linear(v)])  # k,v [b,1,d]
            if cache[layer_name]['k'] is not None:
                k = torch.cat((cache[layer_name]['k'], k), dim=-2)
                v = torch.cat((cache[layer_name]['v'], v), dim=-2)
            cache[layer_name]['k'] = k
            cache[layer_name]['v'] = v
            att_idxs = ~self.mask_block[cur_idx, :cur_idx + 1]
        else:
            if cache[layer_name]['k_cond'] is not None:
                k = cache[layer_name]['k_cond']
                v = cache[layer_name]['v_cond']
            else:
                k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), [self.k_linear(k), self.v_linear(v)])
                cache[layer_name]['k_cond'] = k
                cache[layer_name]['v_cond'] = v
            att_idxs = ~self.mask_block[cur_idx, :]

        att_k = k[:, :, att_idxs, ]
        att_v = v[:, :, att_idxs, ]

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, att_k) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, att_v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = self.dropout(out)
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


class ParallelFeedForward(nn.Module):
    def __init__(self, dim, dropout=0., mult=4.):
        super().__init__()
        # bias for column / row parallel is enabled by default
        self.dense_h_to_8h = nn.Linear(dim, dim * mult * 2)

        self.activation = nn.Sequential(
            GEGLU(),
            nn.Dropout(dropout)
        )
        self.dense_4h_to_h = nn.Linear(dim * mult, dim)

    def forward(self, x):
        out = self.dense_h_to_8h(x)
        out = self.activation(out)
        out = self.dense_4h_to_h(out)
        return out


class SingleTransformer(nn.Module):
    def __init__(self, attention, attention_cond, ff, dim, depth):
        super().__init__()
        self.atten_norm = nn.LayerNorm(dim)
        self.attention = attention
        self.attention_scale = LayerScale(dim, depth)

        if attention_cond is not None:
            self.atten_norm_cond = nn.LayerNorm(dim)
            self.attention_cond = attention_cond
            self.attention_scale_cond = LayerScale(dim, depth)

        self.ff_norm = nn.LayerNorm(dim)
        self.ff = ff
        self.ff_scale = LayerScale(dim, depth)

    def forward(self, x, mask, cond=None, mask_cond=None):
        # attention
        att = self.atten_norm(x)
        att = self.attention(att, att, att, mask)
        att = self.attention_scale(att)
        x = x + att

        # attention_condition
        if cond is not None:
            att = self.atten_norm_cond(x)
            att = self.attention_cond(att, cond, cond, mask_cond)
            att = self.attention_scale_cond(att)
            x = x + att

        # feedforward
        ff = self.ff_norm(x)
        ff = self.ff(ff)
        ff = self.ff_scale(ff)
        ff = x + ff
        return ff

    def forward_with_cache(self, x, cur_idx, mask, cond, mask_cond, layer, cache):
        # attention
        att = self.atten_norm(x)
        att = self.attention.forward_with_cache(att, att, att, cur_idx, mask, False, layer, cache)
        att = self.attention_scale(att)
        x = x + att

        # attention_condition
        if cond is not None:
            att = self.atten_norm_cond(x)
            att = self.attention_cond.forward_with_cache(att, cond, cond, cur_idx, mask_cond, True, layer, cache)
            att = self.attention_scale_cond(att)
            x = x + att

        # feedforward
        ff = self.ff_norm(x)
        ff = self.ff(ff)
        ff = self.ff_scale(ff)
        ff = x + ff
        return ff


class Transformer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            heads=8,
            dim_head=64,
            ff_mult=4,
            attn_dropout=0.,
            ff_dropout=0.,
            causal=True,
            vid_len=None,
            image_fmap_size=None,
            self_attn_types=('full',),
            self_attn_nearby_info=None,
            cond=True,
            cross_attn_types=('full',),
            cross_attn_nearby_info=None,
            use_checkpoint=True,
            **kwargs
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.layers = nn.ModuleList([])

        self_attn_type_layer = islice(cycle(self_attn_types), depth)
        cross_attn_type_layer = islice(cycle(cross_attn_types), depth)

        for ind, self_attn_type, cross_attn_type in zip(range(depth), self_attn_type_layer, cross_attn_type_layer):
            if self_attn_type == 'full':
                self_attn_class = ParallelAttention
            elif self_attn_type == 'nearby':
                self_attn_class = partial(NearbyAttention, sparse_block_info=self_attn_nearby_info,
                                          image_size=image_fmap_size, vid_len=vid_len)
            elif self_attn_type == 'row':
                self_attn_class = partial(AxialAttention, type=0)
            elif self_attn_type == 'col':
                self_attn_class = partial(AxialAttention, type=1)
            else:
                raise ValueError(f'self attention type "{self_attn_type}" is not valid')

            if cross_attn_type == 'full':
                cross_attn_class = ParallelAttention
            elif cross_attn_type == 'nearby':
                cross_attn_class = partial(NearbyAttention, sparse_block_info=cross_attn_nearby_info,
                                           image_size=image_fmap_size, vid_len=vid_len)
            else:
                raise ValueError(f'cross attention type "{self_attn_type}" is not valid')

            self.layers.append(SingleTransformer(
                self_attn_class(dim=dim, causal=causal, heads=heads, dim_head=dim_head, dropout=attn_dropout),
                cross_attn_class(dim=dim, causal=False, heads=heads, dim_head=dim_head,
                                 dropout=attn_dropout) if cond else None,
                ParallelFeedForward(dim, mult=ff_mult, dropout=ff_dropout),
                dim, ind + 1
            ))

    def forward(self, x, mask=None, cond=None, mask_cond=None):
        layers_num = len(self.layers)
        for lid in range(layers_num - 1):
            layer = self.layers[lid]
            if self.use_checkpoint:
                x = checkpoint(layer, x, mask, cond, mask_cond)
            else:
                x = layer(x, mask, cond, mask_cond)
        layer = self.layers[-1]
        return layer(x, mask, cond, mask_cond)

    def forward_with_cache(self, x, cur_idx, mask=None, cond=None, mask_cond=None, incremental_states=None):
        layers_num = len(self.layers)
        for lid in range(layers_num):
            layer = self.layers[lid]
            x = layer.forward_with_cache(x, cur_idx, mask, cond, mask_cond, lid, incremental_states)
        return x


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
