from config.vqvae import *


class Args(BasicArgs):
    task_name, method_name = BasicArgs.parse_config_name(__file__)
    log_dir = os.path.join(BasicArgs.root_dir, task_name, method_name)

    dataset_cf = 'dataset/imagenet.py'
    max_train_samples = None
    max_eval_samples = 64

    train_batch_size = 48
    eval_batch_size = 16
    learning_rate = 4.5e-06
    gradient_accumulate_steps = 1
    epochs = 50
    seed = 42
    num_workers = 16
    eval_step = 1
    save_step = 1

    epoch_threshold = 20
    codebook_weight = 1.0
    discriminator_weight = 0.8
    perceptual_weight = 1.0

    img_size = 256
    coder_config = {'double_z': False, 'z_channels': 256, 'resolution': img_size, 'in_channels': 3, 'out_ch': 3, 'ch': 128,
                    'ch_mult': [1, 1, 2, 2, 4], 'num_res_blocks': 2, 'attn_resolutions': [16], 'dropout': 0.0}

    quantize_config = {'n_e': 16384, 'e_dim': 256, 'beta': 0.25, 'remap': None, 'sane_index_shape': False, 'legacy': False}
    discriminator_config = {'input_nc': 3, 'n_layers': 3, 'use_actnorm': False, 'ndf': 64}
    find_unused_parameters = True
    set_seed(seed)


args = Args()

class Net(nn.Module):
    def __init__(self, args, mode='train'):
        super(Net, self).__init__()
        self.args = args
        self.encoder = Encoder(**args.coder_config)
        self.decoder = Decoder(**args.coder_config)
        self.epoch_threshold = args.epoch_threshold
        self.codebook_weight = args.codebook_weight
        self.discriminator_weight = args.discriminator_weight
        self.perceptual_weight = args.perceptual_weight
        self.quantize = VectorQuantizer(**args.quantize_config)
        self.quant_conv = torch.nn.Conv2d(args.coder_config["z_channels"], args.quantize_config["e_dim"], 1)
        self.post_quant_conv = torch.nn.Conv2d(args.quantize_config["e_dim"], args.coder_config["z_channels"], 1)

        if mode == 'train':
            self.perceptual_loss = LPIPS().eval()
            self.discriminator = NLayerDiscriminator(**args.discriminator_config).apply(weights_init)
            real_learning_rate = args.gradient_accumulate_steps * args.train_batch_size * args.learning_rate
            opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                      list(self.decoder.parameters()) +
                                      list(self.quantize.parameters()) +
                                      list(self.quant_conv.parameters()) +
                                      list(self.post_quant_conv.parameters()),
                                      lr=real_learning_rate, betas=(0.5, 0.9))
            opt_ae.is_enabled = lambda epoch: True
            opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                        lr=real_learning_rate, betas=(0.5, 0.9))
            opt_disc.is_enabled = lambda epoch: epoch >= self.epoch_threshold
            self.optimizers = [opt_ae, opt_disc]
            self.best_metric_fn = lambda x: x['train']['loss_total']
            self.scheduler = None

        elif mode == 'eval':
            self.perceptual_loss = nn.Identity()
            self.discriminator = nn.Identity()

    @torch.no_grad()
    def get_codebook_indices(self, img):
        b = img.shape[0]
        h = self.encoder(img)
        h = self.quant_conv(h)
        _, _, [_, _, indices] = self.quantize(h)
        return rearrange(indices.flatten(), '(b n)-> b n', b=b)

    def decode(self, img_seq, height=None):
        b, n = img_seq.shape
        one_hot_indices = F.one_hot(img_seq, num_classes=self.args.quantize_config['n_e']).float()
        z = (one_hot_indices @ self.quantize.embedding.weight)
        z = rearrange(z, 'b (h w) c -> b c h w', h=height if height else int(math.sqrt(n)))
        quant = self.post_quant_conv(z)

        max_b = 1
        imgs = []
        for q in quant.split(max_b, dim=0):
            img = self.decoder(q)
            imgs.append(img)
        return torch.cat(imgs)

    def forward(self, inputs):
        x = inputs['label_imgs'].to(memory_format=torch.contiguous_format)
        device = x.device
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, qloss, _ = self.quantize(h)
        quant = self.post_quant_conv(quant)
        xrec = self.decoder(quant)

        if self.training and inputs['epoch'] < self.epoch_threshold:
            disc_factor = 0.0
        else:
            disc_factor = 1.0
        outputs = {}
        outputs['logits_imgs'] = xrec
        if not "optimizer_idx" in inputs or inputs['optimizer_idx'] == 0:
            # Reconstruction Loss
            rec_loss = torch.abs(x.contiguous() - xrec.contiguous()).mean()
            # Perceptual Loss
            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(x.contiguous(), xrec.contiguous()).mean()
            else:
                p_loss = torch.tensor([0.0], device=device)
            # NLL Loss
            nll_loss = rec_loss + self.perceptual_weight * p_loss
            last_layer = self.decoder.conv_out.weight

            if disc_factor:
                logits_fake = self.discriminator(xrec.contiguous())
                g_loss = -torch.mean(logits_fake)
                if self.training:
                    nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
                    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
                    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
                    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
                    d_weight = d_weight * self.discriminator_weight
                else:
                    d_weight = torch.tensor(0.0, device=device)
                loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * qloss.mean()
            else:
                g_loss = torch.tensor(0.0, device=device)
                loss = nll_loss + self.codebook_weight * qloss.mean()

            outputs['loss_total'] = loss
            outputs['loss_q'] = qloss
            outputs['loss_rec'] = rec_loss
            outputs['loss_p'] = p_loss
            outputs['loss_nll'] = nll_loss
            outputs['loss_g'] = g_loss

        if not "optimizer_idx" in inputs or inputs['optimizer_idx'] == 1:
            # discriminator
            # TODO Removed detch() for logits_fake
            if disc_factor:
                logits_real = self.discriminator(x.contiguous().detach())
                logits_fake = self.discriminator(xrec.contiguous())
                d_loss = disc_factor * hinge_d_loss(logits_real, logits_fake)
                # d_loss = disc_factor * vanilla_d_loss(logits_real, logits_fake)
            else:
                d_loss = torch.tensor(0.0, device=device)
            outputs['loss_total_1'] = d_loss
            outputs['loss_disc'] = d_loss

        return outputs


