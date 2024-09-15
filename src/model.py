import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention
from einops.layers.torch import Rearrange


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, patch_size, input_ch, n_classes, hidden_dim, threshold=0.35):
        super(TransMIL, self).__init__()
        self.input_ch = input_ch
        self.n_classes = n_classes
        self.threshold = threshold
        patch_dim = input_ch * patch_size
        print(f'model {threshold=}')
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p = patch_size),
            # nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
        )
        self.pos_layer = PPEG(dim=hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=hidden_dim)
        self.layer2 = TransLayer(dim=hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self._fc = nn.Linear(hidden_dim, self.n_classes)


    def forward(self, h):
        h = h.view(1, self.input_ch, -1)
        h = h[:, :, :1024*70] # for now(drop 30 seconds data)

        #---->patch_embedding
        h = self.to_patch_embedding(h) #[B, N-1, hidden_dim]

        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N-1, hidden_dim]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, hidden_dim]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, hidden_dim]

        #---->Translayer x2
        h = self.layer2(h) #[B, N, hidden_dim]

        #---->cls_token
        h = self.norm(h)[:,0]

        #---->predict
        logits = self._fc(h) #[B, n_classes]
        if self.n_classes == 1:
            Y_prob = torch.sigmoid(logits)
            Y_hat = torch.where(Y_prob > self.threshold, 1, 0)
        else:
            Y_prob = F.softmax(logits, dim = 1)
            Y_hat = torch.argmax(Y_prob, dim = 1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict

    def get_embedding(self, h):
        h = h.view(1, self.input_ch, -1)
        h = h[:, :, :1024*70] # for now(drop 30 seconds data)

        #---->patch_embedding
        h = self.to_patch_embedding(h) #[B, N-1, hidden_dim]

        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N-1, hidden_dim]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, hidden_dim]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, hidden_dim]

        #---->Translayer x2
        h = self.layer2(h) #[B, N, hidden_dim]

        #---->cls_token
        h = h[:,0]

        return h