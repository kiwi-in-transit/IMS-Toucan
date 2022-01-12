"""Attention class from lucidrains Performer implementation https://github.com/lucidrains/performer-pytorch"""
import math
import torch
from torch import nn
from torch.cuda.amp import autocast
from einops import rearrange, repeat
from functools import partial
from contextlib import contextmanager
from distutils.version import LooseVersion

TORCH_GE_1_8_0 = LooseVersion(torch.__version__) >= LooseVersion('1.8.0')

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False


def default(val, d):
    return val if exists(val) else d


def exists(val):
    return val is not None


def empty(tensor):
    return tensor.numel() == 0


@contextmanager
def null_context():
    yield


def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device = None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    ratio = (projection_matrix.shape[0] ** -0.5)

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)

    #print('softmax kernel function, projection shape:', projection.shape)
    #print('softmax kernel function, data shape:', data.shape)
    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data -
                    torch.amax(data_dash, dim=-1, keepdim=True)) + eps)
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=(-1, -2), keepdim=True)) + eps)

    return data_dash.type_as(data)


def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling = 0, device = None):
    # default values are 256 / 4, so this would be 64
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device = device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device = device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device = device).norm(dim = 1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device = device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix
    

def orthogonal_matrix_chunk(cols, device = None):
    unstructured_block = torch.randn((cols, cols), device = device)
    if TORCH_GE_1_8_0:
        q, r = torch.linalg.qr(unstructured_block.cpu(), mode = 'reduced')
    else:
        q, r = torch.qr(unstructured_block.cpu(), some = True)
    q, r = map(lambda t: t.to(device), (q, r))
    return q.t()


def causal_linear_attention(q, k, v, eps = 1e-6):
    from fast_transformers.causal_product import CausalDotProduct
    autocast_enabled = torch.is_autocast_enabled()
    is_half = isinstance(q, torch.cuda.HalfTensor)
    assert not is_half or APEX_AVAILABLE, 'half tensors can only be used if nvidia apex is available'
    cuda_context = null_context if not autocast_enabled else partial(autocast, enabled = False)

    causal_dot_product_fn = amp.float_function(CausalDotProduct.apply) if is_half else CausalDotProduct.apply

    k_cumsum = k.cumsum(dim=-2) + eps
    D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q))

    with cuda_context():
        if autocast_enabled:
            q, k, v = map(lambda t: t.float(), (q, k, v))

        out = causal_dot_product_fn(q, k, v)

    out = torch.einsum('...nd,...n->...nd', out, D_inv)
    return out


class FastAttention(nn.Module):

    def __init__(self, dim_heads, nb_features = 256, ortho_scaling = 0, causal = True, generalized_attention = False, kernel_fn = nn.ReLU(), no_projection = False):
        super().__init__()
        # note nb_features is the number of random features for the projection
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        # creates a projection matrix with dimensions [nb_features, dim_heads]
        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows = self.nb_features, nb_columns = self.dim_heads, scaling = ortho_scaling)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.kernel_fn = kernel_fn

        self.causal = causal
        if causal:
            try:
                import fast_transformers.causal_product.causal_product_cuda
                self.causal_linear_fn = partial(causal_linear_attention)
            except ImportError:
                print('unable to import cuda code for auto-regressive Performer')


    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device = device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v):
        device = q.device

        # partial just presets some of the variables of a function, here it presets the projection_matrix and device for the softmax_kernel function
        create_kernel = partial(softmax_kernel, projection_matrix = self.projection_matrix, device = device)
        q = create_kernel(q, is_query = True)
        k = create_kernel(k, is_query = False)

        attn_fn = linear_attention if not self.causal else self.causal_linear_fn
        out = attn_fn(q, k, v)
        
        return out
        

class Attention(nn.Module):
    def __init__(
        self,
        heads,
        dim,
        dropout,
        causal = True,
        #dim_head = 64,
        local_heads = 0,
        local_window_size = 256,
        nb_features = None,
        feature_redraw_interval = 1000,
        generalized_attention = False,
        kernel_fn = nn.ReLU(),
        no_projection = False,
        qkv_bias = False,
        attn_out_bias = True
    ):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        #dim_head = default(dim_head, dim // heads)
        dim_head = dim // heads
        inner_dim = dim_head * heads
        self.fast_attention = FastAttention(dim_head, nb_features, causal = causal, generalized_attention = generalized_attention, kernel_fn = kernel_fn, no_projection = no_projection)

        #taken from PerformerLM class
        self.pos_emb = None
        # also need dim
        self.dim = dim

        self.heads = heads
        self.global_heads = heads - local_heads
        #self.local_attn = LocalAttention(window_size = local_window_size, causal = causal, autopad = True, dropout = dropout, look_forward = int(not causal), rel_pos_emb_config = (dim_head, local_heads)) if local_heads > 0 else None

        self.to_q = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_k = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_v = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim, bias = attn_out_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, _1, _2, mask = None, context_mask = None, pos_emb = None, context = None, **kwargs):

        # add self.pos_emb from PerformerLM class, but only add it once
        #if self.pos_emb is None:
        #    print('x shape:', x.shape)
        #    print('x shape 1:', x.shape[1])
        #    self.pos_emb = FixedPositionalEmbedding(self.dim, x.shape[1])

        #print('x shape, performer forward function:', x.shape)
        b, n, _, h, gh = *x.shape, self.heads, self.global_heads

        #cross_attend = exists(context)

        context = default(context, x)
        #context_mask = default(context_mask, mask) if not cross_attend else context_mask
        #print('mask shape, performer forward function:', mask.shape)
        
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)

	    # split attention_dim into heads and dim, and change the order of the tensor dimensions
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        
        #split q, k, and v into tuples of (up to gh, after gh) for local attention
        #(q, lq), (k, lk), (v, lv) = map(lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))

        attn_outs = []

        #print('v shape, performer forward function', v.shape)
        if not empty(q):
            #if exists(context_mask):
             #   global_mask = context_mask[:, None, :, None]
                 # original mask shape [1, n] --> [1, 1, n, 1]
                 # Error due to Toucan mask shape e.g. [b, n, n] --> [b, 1, n, 1, n]
                 # could just take only the first two dimensions of the Toucan mask..?
             #   v.masked_fill_(~global_mask, 0.)

            # should work without rotary positional embeddings, just not as well. But I don't think this was part of the original performer paper.
            #if exists(self.pos_emb) and not cross_attend:
            #    q, k = apply_rotary_pos_emb(q, k, self.pos_emb)

            out = self.fast_attention(q, k, v)
            attn_outs.append(out)

        # local attention wasn't part of the original performer paper
        #if not empty(lq):
        #    assert not cross_attend, 'local attention is not compatible with cross attention'
        #    out = self.local_attn(lq, lk, lv, input_mask = mask)
        #    attn_outs.append(out)

        out = torch.cat(attn_outs, dim = 1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return self.dropout(out)
