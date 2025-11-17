import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import random
import os
import copy 
import math

class PatchEmbedStem(nn.Module):
    def __init__(self, patch_dim, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Linear(patch_dim, embed_dim)
        self.patch_size  = patch_size

    def patchify_images(self , images):
        patch_size  = self.patch_size
        # images: [B, 3, 32, 32]
        B, C, H, W = images.shape
        n_p = H // patch_size
        patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # patches shape: [B, C, n_p, n_p, patch_size, patch_size]
        patches = patches.contiguous().view(B, C, n_p * n_p, patch_size * patch_size)
        # rearrange to [B, n_patches, patch_dim]
        patches = patches.permute(0, 2, 1, 3).contiguous()   # [B, n_p, C, patch_size*patch_size]
        patches = patches.view(B, n_p * n_p, C * patch_size * patch_size)  # [B, n_patches, patch_dim]
        return patches
    
    def forward(self, x):
        x = self.patchify_images(x)
        out = self.proj(x)
        return out

class Transformer_encoder_layer(nn.Module):
    def __init__(self, d_model, nheads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nheads = nheads
        self.dropout = nn.Dropout(dropout)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.LNM = nn.LayerNorm(d_model)
        self.FFN = nn.Sequential(
        nn.Linear(d_model, d_model * 4),
        nn.ReLU(),
        nn.Linear(d_model * 4, d_model)
        )

    def Multiheadattention(self, q, k, v):
        nheads = self.nheads
        d_model = self.d_model
        d_k = d_model // nheads
        batch_size, seq_len, _ = q.shape

        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)
        Q = Q.view(batch_size, seq_len, nheads, d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, nheads, d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, nheads, d_k).transpose(1, 2)
        Attn_score = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
        Attn_weights = torch.softmax(Attn_score, dim=-1)
        Attention_add = Attn_weights @ V  # (B, h, seq, d_k)
        Attention_add = Attention_add.transpose(1, 2).contiguous()  # (B, seq, h, d_k)
        Attention_add = Attention_add.view(batch_size, seq_len, d_model)  # (B, seq, d_model)
        Attn_out = self.W_o(Attention_add)
        return Attn_out, Attn_weights

    def forward(self, x):
        X_attn, _ = self.Multiheadattention(x, x, x)
        x = x + self.dropout(X_attn)
        x = self.LNM(x)
        # Add FFN here if you want to match a full transformer encoder layer
        residual = x
        x = residual + self.FFN(x)
        x  = self.LNM(x)
        return x

class Transformer_Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        # RIGHT MULTIPLE: make num_layers copies!
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
    def forward(self, x):
        for layer in self.layers:
            x  = layer(x)
        return x




class MiniTransformer(nn.Module):
    def __init__(self, n_patches, embed_dim=64, num_heads=2, num_layers=2 , dropout=0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        encoder_layer = Transformer_encoder_layer(d_model = embed_dim , nheads = num_heads , dropout = dropout)
        self.transformer = Transformer_Encoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    def forward(self, x):
        B = x.size(0)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.transformer(x)
        x = self.norm(x)
        return x[:, 0] 

class Classifier(nn.Module):
    def __init__(self, embed_dim = 64 ,num_classes = 10 ,  dropout=0.1):
        super().__init__()
        self.fc1  = nn.Linear(embed_dim , 2 * embed_dim )
        self.fc2  = nn.Linear(2 * embed_dim , embed_dim)
        self.fc3 = nn.Linear(embed_dim , embed_dim//2)
        self.fc4 = nn.Linear(embed_dim //2 , num_classes)

    def forward(self , x):
        x  = self.fc1(x)
        x  = F.relu(x)
        x  = self.fc2(x)
        x  = F.relu(x)
        x  = self.fc3(x)
        x  = F.relu(x)
        x  = self.fc4(x)
        return x
    
class VIT_classifier(nn.Module):
    def __init__(self,Patchembedding : PatchEmbedStem ,  
                 vit_transformer : MiniTransformer , 
                 classifier: Classifier 
                 ):
        super().__init__()
        self.Patchembedding  = Patchembedding
        self.vit_transformer = vit_transformer
        self.classifier = classifier
    
    def forward(self,x):
        patched_image  = self.Patchembedding(x)
        cls_token = self.vit_transformer(patched_image)
        logits = self.classifier(cls_token)
        return logits
