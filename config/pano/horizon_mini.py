import sys
import torch
sys.path.append('../..')
from config.pano import *
from config.horizon_d_pano_pe import *

# TODO fix train using causal=False
class Args(BasicArgs):
    task_name, method_name = BasicArgs.parse_config_name(__file__)
    log_dir = os.path.join(BasicArgs.root_dir, task_name, method_name)

    # data
    dataset_cf = 'dataset/panohdembd.py'
    max_train_samples = None
    max_eval_samples = None
    max_video_len = 1
    img_size = 256
    img_full_size_h = 768
    img_full_size_w = 1536
    mem_expand_size = {'h': 2, 'w': 2, 't': 0}   # h,w,t
    mem_expand_size_pano = {'h': 1, 'w': 1, 't': 0}

    # vae
    vae_cf = 'config/vqvae/VQGan16384F16.py'
    vae = import_filename(vae_cf)
    VQVAE, args_vqvae = vae.Net, vae.args
    vae_path = os.path.join(BasicArgs.root_dir, 'vqg/VQGan16384F16.pth')
    vqvae_vocab_size = 16384
    compress_ratio = 16

    # model
    dim = 1280
    depth = 24
    heads = 20
    dim_head = 64
    one_kv_head = False
    ff_mult = 4
    attn_dropout = 0.1
    ff_dropout = 0.1
    self_attn_types = ('full',)
    causal = False
    cond = True
    cross_attn_types = ('full',)
    # clip_path = os.path.join(BasicArgs.root_dir, "CLIP", "ViT-B-32.pt")
    clip_path = "ViT-B/32"
    use_checkpoint = False
    find_unused_parameters = True
    gamma_mode = 'cosine'

    # iterative
    iterative_model_class = IterativeModel
    get_pos_from_idx = partial(get_pos_from_idx_down_right)
    get_idx_from_pos = partial(get_idx_from_pos_down_right)

    # training
    train_batch_size = 8
    eval_batch_size = 8
    learning_rate = 1e-4
    warmup_ratio = 0.05
    label_smooth = 0.1
    max_norm = 1.0
    epochs = 100
    num_workers = 32
    eval_step = 3
    save_step = 3

    # sample
    tk = 0  # How many top logits we consider to sample.
    tp = 1
    step = 12
    temperature_start = 3.0
    temperature_end = 1.0
    temperature_anneal_mode = 'linear'

    sample_k = 1  # How many times we sample
    best_n = 1  # How many times we visu for sample_K.

    img_gen_height = 768
    img_gen_width = 1536
    img_gen_frame = 1
    init_patch_num = 0

    # others
    seed = 42
    set_seed(seed)

args = Args()


class ClipTextEncoder(nn.Module):
    def __init__(self, model_path, dim):
        super(ClipTextEncoder, self).__init__()
        model, _ = clip.load(model_path, device='cpu')
        self.token_embedding = copy.deepcopy(model.token_embedding)
        self.positional_embedding = copy.deepcopy(model.positional_embedding)
        self.transformer = copy.deepcopy(model.transformer)
        self.ln_final = copy.deepcopy(model.ln_final)
        self.cond_emb = nn.Linear(512, dim)

        self.freeze_net_without_last_layer(self.cond_emb)

    def freeze_net_without_last_layer(self, last_layer):
        for n, p in self.named_parameters():
            p.requires_grad = False

        for n, p in last_layer.named_parameters():
            p.requires_grad = True

    def forward(self, cond):
        cond = self.token_embedding(cond)  # [batch_size, n_ctx, d_model]
        cond = cond + self.positional_embedding
        cond = cond.permute(1, 0, 2)  # NLD -> LND
        cond = self.transformer(cond)
        cond = cond.permute(1, 0, 2)  # LND -> NLD
        cond = self.ln_final(cond)
        output = self.cond_emb(cond)  # 512 -> dim
        return output


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.vae = args.VQVAE(args.args_vqvae, mode='eval')
        self.vae.load_state_dict(file2data(args.vae_path, map_location='cpu'), strict=False)
        for n, p in self.vae.named_parameters():
            p.requires_grad = False

        dim = self.args.dim
        self.mem_expand_size = args.mem_expand_size   # {'h': 2, 'w': 2, 't': 3}
        self.mem_expand_size_pano = args.mem_expand_size_pano   # {'h': 2, 'w': 2, 't': 3}
        self.vocab_size = self.args.vqvae_vocab_size
        self.encoder = ClipTextEncoder(model_path=self.args.clip_path, dim=dim)
        self.hint_emb = nn.Linear(512, dim)

        self.img_h = max(1, self.args.img_size // self.args.compress_ratio)  # 256//16=16
        self.img_w = max(1, self.args.img_size // self.args.compress_ratio)
        self.img_t = 1
        self.image_seq_len = self.img_t * self.img_h * self.img_w  # 1*16*16=256

        img_full_h = self.args.img_full_size_h // self.args.compress_ratio  # 1024//16=64
        img_full_w = self.args.img_full_size_w // self.args.compress_ratio
        img_full_t = self.args.max_video_len

        self.patch_h = img_full_h // self.img_h  # 64//16=4
        self.patch_w = img_full_w // self.img_w
        self.patch_t = img_full_t // self.img_t  # 1//1=1
        self.patch_seq_len = self.patch_t * self.patch_h * self.patch_w  # 1*4*4=16

        self.vae_emb = nn.Embedding.from_pretrained(copy.deepcopy(self.vae.quantize.embedding.weight), freeze=True)
        self.image_emb = nn.Linear(self.vae_emb.embedding_dim, dim)
        self.image_mask_emb = nn.Parameter(torch.randn(dim))  # mask embedding of image
        self.image_pos_emb = AxialPositionalEmbedding(dim, axial_shape=(self.img_t, self.img_h, self.img_w))

        self.transformer = Transformer(dim=dim, depth=args.depth, heads=args.heads, dim_head=args.dim_head,
                                       one_kv_head=args.one_kv_head, ff_mult=args.ff_mult,
                                       attn_dropout=args.attn_dropout, ff_dropout=args.ff_dropout,
                                       seq_len=self.image_seq_len,
                                       seq_frame=self.img_t, seq_h=self.img_h, seq_w=self.img_w,
                                       mem_expand_size=args.mem_expand_size,
                                       causal=args.causal, self_attn_types=args.self_attn_types,
                                       cond=args.cond, cross_attn_types=args.cross_attn_types,
                                       use_checkpoint=args.use_checkpoint, img_full_h=img_full_h, img_full_w=img_full_w)

        self.to_logits_img = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.vocab_size),
        )

        self.CE_Smooth = LabelSmoothing(smoothing=args.label_smooth)
        self.gamma = gamma_func(self.args.gamma_mode)

    def get_codebook_indices(self, image):
        b, l, c, h, w = image.size()
        image = image.reshape(-1, c, h, w)  # [b*l,3,1024,1024]
        h_token, w_token = h // self.args.compress_ratio, w // self.args.compress_ratio
        with torch.no_grad():
            image_tokens = self.vae.get_codebook_indices(image).reshape(b, l, h_token, w_token)
        return image_tokens

    def train(self, *args):
        super().train(*args)
        self.vae.eval()

    def forward(self, inputs):
        outputs = dict()

        if self.training:
            outputs = self.forward_(inputs, outputs)
        else:
            outputs = self.sample(inputs, outputs)

        return outputs

    def forward_(self, inputs, outputs=dict()):
        device = self.vae_emb.weight.device
        patch_idx = inputs['patch_idx']   # int
        patch = inputs['patch']     # [b,n]
        memory = inputs['memory']   # [b,p*n,d]

        text = clip.tokenize(inputs['input_text'], truncate=True).to(device)
        mask_cond = text.eq(0)  # [b,n_text]
        cond_emb = self.encoder(text)  # [b,n_text,d]
                # take image hints as cond
        img_hints = self.hint_emb(inputs['input_imghint_feats'])
        mask_hints = torch.full(img_hints.shape[:-1], False, dtype=torch.bool, device=device)
        # print(f"mask hints: {mask_hints.shape}")
        # print(f"img hints: {img_hints.shape}")
        cond_emb = img_hints
        mask_cond = mask_hints

        if exists(memory.mems):
            cond_emb = torch.cat([memory.mems, cond_emb], dim=-2)  # [b,p*n+n_text,d]
            mask_cond = F.pad(mask_cond, (memory.mems.shape[-2], 0), value=False)

        b, n = patch.shape
        image_emb = self.vae_emb(patch)  # [b,n] - > [b,n,d]
        image_emb = self.image_emb(image_emb)

        r = self.gamma(np.random.uniform())
        mask_map = torch.bernoulli(r * torch.ones(patch.shape, device=device)).reshape(*patch.shape, 1)
        mask_emb = repeat(self.image_mask_emb, 'd -> b n d', b=b, n=n)
        x_emb = image_emb * (1-mask_map) + mask_emb * mask_map
        x_emb += self.image_pos_emb(x_emb)

        # out:[b,n,d]
        out = self.transformer(x=x_emb, mask=None, cond=cond_emb, mask_cond=mask_cond,
                               patch_info=(self.patch_h, self.patch_w), cur_patch_idx=patch_idx,
                               mem_idxs=memory.idxs, locate_fn=args.get_pos_from_idx)
        logits_seq = self.to_logits_img(out)

        # mask_idxs:[num,2(b,n)]
        mask_idxs = mask_map.reshape(patch.shape).nonzero(as_tuple=False)

        # logits_seq:[b,n,c]->[num,c]  image_tokens:[b,n]->[num]
        logits_seq = logits_seq[mask_idxs[:, 0], mask_idxs[:, 1]].reshape(-1, self.vocab_size)
        gt_tokens = patch[mask_idxs[:, 0], mask_idxs[:, 1]].reshape(-1)

        if gt_tokens.shape[0] > 0:
            loss_img = self.CE_Smooth(logits_seq, gt_tokens)
        else:
            loss_img = torch.zeros(1, requires_grad=True, **to(image_emb))

        outputs['new_mem'] = (image_emb + self.image_pos_emb(image_emb)).detach()
        outputs['loss_img'] = loss_img
        outputs['loss_total'] = loss_img

        return outputs

    @torch.no_grad()
    def sample(self, inputs, outputs=dict()):
        b, k, device = inputs['label_imgs'].shape[0], self.args.sample_k, self.vae_emb.weight.device
        n = self.image_seq_len
        B = b*k

        img_gen_full_h = self.args.img_gen_height // self.args.compress_ratio  # 1024/16=64
        img_gen_full_w = self.args.img_gen_width // self.args.compress_ratio   # 2048/16=128
        img_gen_full_t = self.args.img_gen_frame

        patch_gen_h = img_gen_full_h // self.img_h  # 64//16=4
        patch_gen_w = img_gen_full_w // self.img_w  # 128//16=8
        patch_gen_t = img_gen_full_t // self.img_t  # 1//1=1
        patch_gen_seq_len = patch_gen_h * patch_gen_w * patch_gen_t  # 1*4*8=32

        get_pos_from_idx = partial(args.get_pos_from_idx, patch_h=patch_gen_h, patch_w=patch_gen_w)
        get_idx_from_pos = partial(args.get_idx_from_pos, patch_h=patch_gen_h, patch_w=patch_gen_w)

        # memory pool
        mem_pool = dict()
        mem_mapping, max_used_mapping = get_mem_mapping(img_patch_h=patch_gen_h, img_patch_w=patch_gen_w,
                                                        img_patch_t=patch_gen_t, mem_expand_size=self.mem_expand_size,
                                                        get_pos_from_idx=get_pos_from_idx,
                                                        get_idx_from_pos=get_idx_from_pos)

        mem_mapping_pano, max_used_mapping_pano = get_mem_mapping_pano(img_patch_h=patch_gen_h, img_patch_w=patch_gen_w,
                                                        img_patch_t=patch_gen_t, mem_expand_size=self.mem_expand_size_pano,
                                                        get_pos_from_idx=get_pos_from_idx,
                                                        get_idx_from_pos=get_idx_from_pos)

        text = clip.tokenize(inputs['input_text'], truncate=True).to(device)
        text = repeat(text, 'b n -> (b k) n', k=k)
        mask_text_cond = text.eq(0)
        cond_text_emb = self.encoder(text)  # [b,77,d]
        img_hints = self.hint_emb(inputs['input_imghint_feats'])
        mask_hints = torch.full(img_hints.shape[:-1], False, dtype=torch.bool, device=device)
        # print(f"mask hints: {mask_hints.shape}")
        # print(f"img hints: {img_hints.shape}")
        cond_text_emb = img_hints
        mask_text_cond = mask_hints

        _, l, c, h_img, w_img = inputs['label_imgs'].size()
        h_token, w_token = h_img // self.args.compress_ratio, w_img // self.args.compress_ratio
        image = inputs['label_imgs'].reshape(-1, c, h_img, w_img)
        # [b,1,64,64]
        image = self.vae.get_codebook_indices(image).reshape(b, l, h_token, w_token)

        # down-right  [p,b,16*16]
        gt_tokens = rearrange(image, 'b l (ph h) (pw w) -> (l pw ph) b (h w)', ph=self.patch_h, pw=self.patch_w)

        seed = torch.seed()
        os.environ['PYHTONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        init_patch_num = self.args.init_patch_num
        step = self.args.step
        gamma = gamma_func(self.args.gamma_mode)
        mask_emb = repeat(self.image_mask_emb, 'd -> b n d', b=B, n=n)

        full_token_indices_first_pass = self.infer(k, device, n, B, patch_gen_h, patch_gen_w, patch_gen_seq_len, get_pos_from_idx, mem_pool, mem_mapping, max_used_mapping, mask_text_cond, cond_text_emb, gt_tokens, init_patch_num, step, gamma, mask_emb, clean = False)
        full_token_indices_sec_pass = self.infer(k, device, n, B, patch_gen_h, patch_gen_w, patch_gen_seq_len, get_pos_from_idx, mem_pool, mem_mapping_pano, max_used_mapping_pano, mask_text_cond, cond_text_emb, gt_tokens, init_patch_num, step, gamma, mask_emb, clean = True)
        
        gen_images_first = self.decode_img(full_token_indices_first_pass, b, k, img_gen_full_h, patch_gen_h, patch_gen_w)
        gen_images_second = self.decode_img(full_token_indices_sec_pass, b, k, img_gen_full_h, patch_gen_h, patch_gen_w)
        outputs['logits_imgs'] = torch.cat([gen_images_first, gen_images_second], dim=1)

        return outputs

    def decode_img(self, full_token_indices, b, k, img_gen_full_h, patch_gen_h, patch_gen_w):
        full_token_indices = torch.stack(full_token_indices, dim=0)
        full_token_indices = rearrange(full_token_indices, '(l pw ph) b h w -> (b l) (ph h) (pw w)',
                                       ph=patch_gen_h, pw=patch_gen_w)
        img_seq_full = full_token_indices
        gen_images_shift_list = []
        d = self.img_w * 2           
        img_seq_shift = torch.cat([img_seq_full[:,:,-d:],img_seq_full, img_seq_full[:,:,:d]], dim=-1)
        img_seq_shift = img_seq_shift.reshape(int(b * k * self.args.img_gen_frame), -1)
        images_shift = self.vae.decode(img_seq_shift, height=img_gen_full_h)
        gen_images_shift = rearrange(images_shift, '(b k l) c h w -> b k l c h w', b=b, k=k)
        gen_images_shift = gen_images_shift[:, :, :, :, :, d*self.args.compress_ratio:-d*self.args.compress_ratio]
        gen_images_shift_list.append(gen_images_shift)
        
        full_token_indices = full_token_indices.reshape(int(b * k * self.args.img_gen_frame), -1)
        images = self.vae.decode(full_token_indices, height=img_gen_full_h)
        gen_images = rearrange(images, '(b k l) c h w -> b k l c h w', b=b, k=k)
        gen_images_shift_list.append(gen_images)
        gen_images = torch.cat(gen_images_shift_list, dim=1)
        return gen_images

    def infer(self, k, device, n, B, patch_gen_h, patch_gen_w, patch_gen_seq_len, get_pos_from_idx, mem_pool, mem_mapping, max_used_mapping, mask_text_cond, cond_text_emb, gt_tokens, init_patch_num, step, gamma, mask_emb, clean):
        # inference
        full_token_indices = []
        for patch_idx in tqdm(range(patch_gen_seq_len)):
            if patch_idx < init_patch_num:
                token_indices = repeat(gt_tokens[patch_idx], 'b n -> (b k) n', k=k)  # [b*k,n]
            else:
                # [b,n]  init token_indices = -1
                token_indices = torch.full((B, n), fill_value=-1, dtype=torch.int64, device=device)
                memory = get_mem_from_pool(cur_patch_idx=patch_idx, mem_pool=mem_pool, mem_mapping=mem_mapping)
                if exists(memory.mems):
                    cond_emb = torch.cat([memory.mems, cond_text_emb], dim=-2)  # [b,p*n+n_text,d]
                    mask_cond = F.pad(mask_text_cond, (memory.mems.shape[-2], 0), value=False)
                else:
                    cond_emb = cond_text_emb
                    mask_cond = mask_text_cond

                for step_idx in range(step):
                    ret_count_prev = n - int(n * gamma(step_idx / step))
                    ret_count = n - int(n * gamma((step_idx+1) / step))

                    mask_map = (token_indices < 0).type(torch.int).reshape(B, n, 1)

                    image_emb = self.image_emb(self.vae_emb(torch.clamp(token_indices, min=0)))
                    image_emb = image_emb * (1-mask_map) + mask_emb * mask_map
                    image_emb += self.image_pos_emb(image_emb)
                    out = self.transformer(x=image_emb, mask=None, cond=cond_emb, mask_cond=mask_cond,
                                           patch_info=(patch_gen_h, patch_gen_w), cur_patch_idx=patch_idx,
                                           mem_idxs=memory.idxs, locate_fn=get_pos_from_idx)
                    logits = self.to_logits_img(out).reshape(-1, self.vocab_size)  # [b*n,c]

                    # filtered_logits = top_k_top_p_filtering(logits, top_k=self.args.tk, top_p=self.args.tp)
                    filtered_logits = logits
                    probs = F.softmax(filtered_logits, dim=-1)  # [b*n,c]
                    sample = torch.multinomial(probs, 1)  # [b*n]
                    sample_probs = torch.gather(probs.reshape(B, n, -1), -1, sample.reshape(B, n, 1)).reshape(B, n)  # [b,n]

                    # retain token prob will be set 1.0
                    max_v = max_pos_value(sample_probs)
                    sample_probs = torch.where(token_indices >= 0, torch.full_like(sample_probs, fill_value=max_v), sample_probs)  # [b,n]

                    # top_k token will be retained
                    temperature = temperature_anneal(t=(step_idx+1) / step, anneal_mode=self.args.temperature_anneal_mode,
                                                     start=self.args.temperature_start, end=self.args.temperature_end)

                    confidence = torch.log(sample_probs) + temperature * -torch.empty_like(sample_probs).exponential_().log()  # [b,n]

                    _, idxs_sorted = torch.sort(confidence, dim=-1, descending=True)  # [b,n]
                    idxs_ret_increase = idxs_sorted[:, ret_count_prev:ret_count]  # [b,ret_count_increase]
                    token_indices.scatter_(-1, idxs_ret_increase, torch.gather(sample.reshape(B, n), -1, idxs_ret_increase))

            image_emb = self.image_emb(self.vae_emb(token_indices))
            update_mem(cur_patch_idx=patch_idx, mem_pool=mem_pool, max_used_mapping=max_used_mapping,
                       new_mem=(image_emb + self.image_pos_emb(image_emb)).detach(), clean=clean)
            full_token_indices.append(token_indices.reshape(B, self.img_h, self.img_w))
        return full_token_indices
