"""Seven-depth-checkpoint differentiable gated multimodal student (PyTorch optional)."""
import torch
from torch import nn

class MaskedStudent(nn.Module):
    """Input features from frozen encoders -> learned depth gates -> class logits."""
    def __init__(self, input_dim, n_classes, width=256, depth=14,
                 checkpoints=7, mode="shared", dropout=0.0):
        super().__init__()
        if depth < checkpoints or checkpoints < 1 or mode not in ("none","single","shared","independent"):
            raise ValueError("invalid depth/checkpoint/mode")
        self.project=nn.Linear(input_dim,width)
        self.blocks=nn.ModuleList([nn.Sequential(nn.LayerNorm(width),
            nn.Linear(width,width),nn.GELU(),nn.Dropout(dropout),
            nn.Linear(width,width)) for _ in range(depth)])
        self.positions=set((i+1)*depth//checkpoints-1 for i in range(checkpoints))
        self.mode=mode
        self.checkpoints=checkpoints
        if mode=="shared":
            self.depth_embedding=nn.Embedding(checkpoints,16)
            self.gate=nn.Sequential(nn.Linear(width+16,width//2),
                                    nn.GELU(),nn.Linear(width//2,width))
        elif mode=="independent":
            self.gates=nn.ModuleList([nn.Sequential(nn.Linear(width,width//2),
                   nn.GELU(),nn.Linear(width//2,width)) for _ in range(checkpoints)])
        elif mode=="single":
            self.gate=nn.Linear(width,width)
        self.norm=nn.LayerNorm(width)
        self.head=nn.Linear(width,n_classes)

    def forward(self,x,return_masks=False):
        x=self.project(x)
        masks=[]
        idx=0
        for i,block in enumerate(self.blocks):
            change=block(x)
            if i in self.positions and self.mode!="none":
                if self.mode=="independent":
                    m=torch.sigmoid(self.gates[idx](x))
                elif self.mode=="shared":
                    pos=self.depth_embedding.weight[idx].expand(x.shape[0],-1)
                    m=torch.sigmoid(self.gate(torch.cat([x,pos],dim=-1)))
                else:
                    m=torch.sigmoid(self.gate(x)) if idx==0 else torch.ones_like(x)
                change=change*m
                masks.append(m)
                idx+=1
            x=x+change
        logits=self.head(self.norm(x))
        if return_masks:
            return logits,masks
        return logits

def mask_penalty(masks):
    """Mean gate activation (add weighted term to the training loss)."""
    if not masks: return None
    return torch.stack([m.mean() for m in masks]).mean()
