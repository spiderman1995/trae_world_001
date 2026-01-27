import torch
from fvcore.nn import FlopCountAnalysis

from vit_model import Attention

# 计算并对比自注意力（Self-Attention）和多头注意力（Multi-Head Attention）的计算量（以 FLOPs 为单位），用于分析两种注意力机制的计算复杂度差异。
def main():
    # Self-Attention
    a1 = Attention(dim=512, num_heads=1)
    a1.proj = torch.nn.Identity()  # remove Wo

    # Multi-Head Attention
    a2 = Attention(dim=512, num_heads=8)

    # [batch_size, num_tokens, total_embed_dim]
    t = (torch.rand(32, 1024, 512),)

    flops1 = FlopCountAnalysis(a1, t)
    print("Self-Attention FLOPs:", flops1.total())

    flops2 = FlopCountAnalysis(a2, t)
    print("Multi-Head Attention FLOPs:", flops2.total())


if __name__ == '__main__':
    main()

