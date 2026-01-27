"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    随机深度
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    in_c:rbg = 3
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # 224/16=14
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):        
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."  # 如果传入的高和宽和我们预先设定的不一样的话，报错

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)  # 展平处理，从第二个维度。.transpose 将维度一二上的顺序进行调换
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,  #生成qkv的时候是否使用偏置，默认false
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  #每个qkv head对应的dim
        self.scale = qk_scale or head_dim ** -0.5  # head_dim的-0.5次方 = 1/sqrt(head_dim)   Attention(Q,K,V) = softmax(QK^T/sqrt(dk))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # 全连接层。别人的源码有的是通过三个分别得到qkv的线性层，这里是通过一个线性层得到的  。没有区别
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)    # 多头拼接后会通过Wo线性变换(映射)，也可以用全连接层实现
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]   ，batch_size 这批图片数目；num_patches 14*14=196个patch；+1是因为加了cls_token；total_embed_dim是每个patch映射后的维度，比如768，vit-b是768
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]  这样调整方便后续计算
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)  切片的方式拿到qkv的数据

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]

        # =====================================================
        # 1 attn = (q @ k.transpose(-2, -1)) * self.scale  #就是对它进行一个Norm处理 
        # 2 attn = attn.softmax(dim=-1)             # 结果的每一行 进行一个softmax处理
        # 3 attn = self.attn_drop(attn)

        # # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        # 4 x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # 。矩阵相乘， .reshape就是把最后两个维度的信息（num_heads, embed_dim_per_head）拼接在一起
        # =========================这1234行替换成以下2行代码=====
        # 自动调用 FlashAttention 或 MemoryEfficientAttention
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0., # 仅在训练时启用 Dropout
            scale=self.scale
        )
        # 恢复维度: [B, num_heads, N, head_dim] -> [B, N, num_heads, head_dim] -> [B, N, C]
        x = x.transpose(1, 2).reshape(B, N, C)

        


        x = self.proj(x) # 全连接层 映射
        x = self.proj_drop(x) # dropout
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        # in_features 输入节点个数  ，hidden_features 第一个全连接层的节点个数，一般是in_features的4倍，
        # out_features 第二个全连接层的节点个数，一般和in_features一样
        # act_layer 激活函数，默认是GELU
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        # 实例化多头注意力模块
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))  # 依次经过 归一化、多头注意力、drop_path，然后和输入x相加。前半段encoder的残差连接
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # 依次经过 归一化、MLP、drop_path，然后和输入x相加。后半段encoder的残差连接    
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer  指的是重复堆叠encoder block的次数
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set 对应最后mlphead中的prelogits 全连接层的节点个数。如果是none,就不会构建对应最后mlphead中的prelogits
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer  对应patch embedding的模块,一般是PatchEmbed类.默认值就是PatchEmbed
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1  # distilled不用管，所以self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  #直接使用一个0矩阵进行初始化 1 batch维度，不用管，方便contact拼接才加上 。1*768
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None #默认none
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        # 是创建一个由多个Transformer编码器块（Block）组成的序列模型（nn.Sequential），
        # 这些编码器块将被堆叠在一起，形成Transformer模型的核心部分。
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None  #和vit无关
        if distilled:           #和vit无关
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        # 特征提取的核心流程，负责将图像 patch 的嵌入特征经过位置编码、Transformer 编码器块加工后，提取出用于分类的关键特征
        # 将 patch 嵌入特征（x）与位置编码（self.pos_embed）相加，再通过 dropout 层
        x = self.pos_drop(x + self.pos_embed)
        # 将添加位置编码后的特征输入到堆叠的 Transformer 编码器块（self.blocks）
        x = self.blocks(x)
        # 对 Transformer 编码器块输出的特征进行层归一化（LayerNorm）
        x = self.norm(x)
        # 默认情况，非蒸馏模式
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])  #切片形式也是取到最前面的
        else:
            return x[:, 0], x[:, 1]
    # ViT）的前向传播核心逻辑，负责将输入图像经过特征提取后，最终输出分类预测结果。
    def forward(self, x):
        # 调用forward_features方法对输入图像x进行特征提取。 
        # 过程包括：图像分块嵌入（patch embed）→ 加入 cls_token 和位置编码 → 通过多层 Transformer 编码器 → 提取 cls_token 的特征。
        # 输出的x是经过 Transformer 加工后的关键特征（非蒸馏模式下是cls_token的特征；蒸馏模式下是cls_token和dist_token的特征 tuple）。
        x = self.forward_features(x)
        # self.head_dist是蒸馏模式下额外的分类头（用于辅助训练），此时x是一个 tuple (cls_token特征, dist_token特征)。
        if self.head_dist is not None:
            # 用主分类头self.head对cls_token特征（x[0]）做预测，得到x（主分类结果）。
            # 用蒸馏分类头self.head_dist对dist_token特征（x[1]）做预测，得到x_dist（蒸馏辅助结果）。
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            # 训练时（self.training为 True）：返回两个预测结果(x, x_dist)，用于计算蒸馏损失（主分类损失 + 辅助损失）。
            # 推理时：返回两个结果的平均值(x + x_dist)/2，作为最终预测（融合两个头的信息，提升精度）
            # torch.jit.is_scripting()： 用于判断当前代码是否处于 TorchScript 脚本化编译环境的函数，返回一个布尔值（True 或 False）。
            if self.training and not torch.jit.is_scripting(): 
                # during inference, return the average of both classifier predictions
                return x, x_dist # # 训练时且非脚本编译，返回两个预测结果
            else:
                return (x + x_dist) / 2 # 推理时或脚本编译时，返回融合结果
        else:
            x = self.head(x)  #对应最后的全连接层linear
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    # 判断传入的模块 m 是否是 nn.Linear 类型
    if isinstance(m, nn.Linear):
        # 如果是 nn.Linear，使用截断正态分布初始化权重，标准差为 0.01 。 避免极端值、保持分布特性
        nn.init.trunc_normal_(m.weight, std=.01)
        # 如果该线性层有偏置项，将偏置项初始化为 0
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    # 判断传入的模块 m 是否是 nn.Conv2d 类型
    elif isinstance(m, nn.Conv2d):
        # 如果是 nn.Conv2d，使用 He 初始化方法初始化权重，mode="fan_out" 表示按输出特征图的维度进行归一化
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        # 如果该卷积层有偏置项，将偏置项初始化为 0
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    # 判断传入的模块 m 是否是 nn.LayerNorm 类型 ，LayerNorm（层归一化）
    elif isinstance(m, nn.LayerNorm):
        # 如果是 nn.LayerNorm，将偏置项初始化为 0
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    # vit模型分为三类 base、large、huge
    """
    # depth=12 表示模型中有 12 个 Transformer 编码器块
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16, 
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1hCv0U8pQomwAtHBYc4hmZg  密码: s5hl
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,   # 16 的计算量更大，是32的4倍
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224(num_classes: int = 1000):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1cxBgZJJ6qUWPSBNcE4TdRQ  密码: qqt8
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    作者从官方权重转换得到的pytorch 预训练权重。不使用预训练权重，效果很差。只有在非常大的数据集上预训练后才会有比较好的效果

    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model
