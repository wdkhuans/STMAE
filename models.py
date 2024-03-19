import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import Block

class STMAE_Pre(nn.Module):
    def __init__(self, embed_dim=512, depth=6, num_heads=4,
                 decoder_embed_dim=256, decoder_depth=2, decoder_num_heads=4,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 node_dim=6, window_size=150, node_num=7, mask_ratio=0.5, len_mask=1):
        super().__init__()

        self.len_mask = len_mask
        self.mask_ratio = mask_ratio
        self.window_size = window_size
        self.node_num = node_num
        self.conv1 = nn.Conv2d(node_dim, embed_dim, kernel_size=(self.node_num, 5), stride=1, padding=(0, 2))
        self.bn1 = nn.BatchNorm2d(embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, window_size + 1, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, window_size + 1, decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, node_dim*node_num, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_ts_sincos_pos_embed(self.pos_embed.shape[-1], int(self.window_size), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        decoder_pos_embed = get_ts_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.window_size), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        w = self.conv1.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device) 
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0 
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore 
    
    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1) 
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        # add pos embed
        x = x + self.decoder_pos_embed
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        # predictor projection
        x = self.decoder_pred(x)
        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward(self, imgs): 
        imgs = imgs.reshape(imgs.shape[0], imgs.shape[1], self.node_num, -1)
        imgs = imgs.permute(0, 2, 1 ,3)

        mask = torch.zeros(imgs.shape[0], imgs.shape[1], dtype=torch.bool).to(imgs.device)
        noise = torch.rand(imgs.shape[0], imgs.shape[1]).to(imgs.device)
        _, indices = torch.topk(noise, self.len_mask, dim=1)
        mask.scatter_(1, indices, True)
        mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).expand_as(imgs)
        imgs[mask_expanded] = 0

        x = imgs.permute(0, 3, 1, 2)
        x = self.bn1(self.conv1(x))
        x = x.squeeze()
        x = x.permute(0, 2, 1)

        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio=self.mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        imgs = imgs.permute(0, 2, 1, 3)
        imgs = imgs.reshape(imgs.shape[0], imgs.shape[1], -1)

        idx = mask.nonzero()
        pred = pred[idx[:, 0], idx[:, 1], :]
        imgs = imgs[idx[:, 0], idx[:, 1], :]
        pred = pred.reshape(x.shape[0], -1, pred.shape[1])
        imgs = imgs.reshape(x.shape[0], -1, imgs.shape[1])

        return imgs, pred

class STMAE_Finetune(nn.Module):
    def __init__(self, embed_dim=512, depth=6, num_heads=4,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 node_dim=6, window_size=150, node_num=7, num_classes=8):
        super().__init__()
        
        self.head = nn.Linear(embed_dim, num_classes)

        self.window_size = window_size
        self.node_num = node_num
        self.conv1 = nn.Conv2d(node_dim, embed_dim, kernel_size=(self.node_num, 5), stride=1, padding=(0, 2))
        self.bn1 = nn.BatchNorm2d(embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, window_size + 1, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_ts_sincos_pos_embed(self.pos_embed.shape[-1], int(self.window_size), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.conv1.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder_full(self, x):

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward(self, imgs):
        imgs = imgs.reshape(imgs.shape[0], imgs.shape[1], self.node_num, -1)
        imgs = imgs.permute(0, 2, 1 ,3) 

        x = imgs.permute(0, 3, 1, 2)
        x = self.bn1(self.conv1(x))
        x = x.squeeze()
        x = x.permute(0, 2, 1)
        x = self.forward_encoder_full(x)
        x= x.mean(dim=1)
        x = self.head(x)

        return x
    
def get_ts_sincos_pos_embed(embed_dim, window_size, cls_token=False):
    """
    embed_dim: output dimension for each position
    window_size: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = np.arange(window_size, dtype=np.float32)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    pos_embed = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0) # (M+1, D)
    return pos_embed
    
def fetch_classifier(method, args=None):
    if 'STMAE_Pre' in method:
        model = STMAE_Pre(embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio, 
        norm_layer=nn.LayerNorm, node_dim=args.dataset_cfg.node_dim, window_size=args.dataset_cfg.seq_len, node_num=args.dataset_cfg.node_num,
        decoder_embed_dim=args.decoder_embed_dim, decoder_depth=args.decoder_depth, decoder_num_heads=args.decoder_num_heads, 
        mask_ratio=args.mask_ratio, len_mask=args.len_mask)
    elif 'STMAE_Finetune' in method:
        model = STMAE_Finetune(embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio,
        norm_layer=nn.LayerNorm, node_dim=args.dataset_cfg.node_dim, window_size=args.dataset_cfg.seq_len, node_num=args.dataset_cfg.node_num, num_classes=args.dataset_cfg.activity_label_size)
    else:
        model = None
    return model
