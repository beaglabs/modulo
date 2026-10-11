"""Optional trainable multimodal fusion backbone (PyTorch extra)."""
import torch
from torch import nn
from torch.nn import functional as F

class FusionBackbone(nn.Module):
    """Tokenized pretrained embeddings and numeric arrays, dynamic per-modality adapters."""
    def __init__(self, inputs, width=640, depth=12, heads=10, max_tokens=128, patch=32):
        super().__init__()
        if width%heads: raise ValueError("width must be divisible by heads")
        self.inputs=dict(inputs)
        self.patch=patch;self.max_tokens=max_tokens
        self.projections=nn.ModuleDict({key:nn.Linear(patch if size is None else size,width) for key,size in inputs.items()})
        self.type_embeddings=nn.ParameterDict({key:nn.Parameter(torch.randn(width)*.01) for key in inputs})
        self.cls=nn.Parameter(torch.zeros(1,1,width))
        self.pos=nn.Parameter(torch.randn(1,max_tokens,width)*.01)
        layer=nn.TransformerEncoderLayer(width,heads,4*width,batch_first=True,norm_first=True)
        self.body=nn.TransformerEncoder(layer,depth,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(width)

    def forward(self, observations):
        outputs=[]
        for obs in observations:
            chunks=[self.cls[0]]
            for key,size in self.inputs.items():
                if key not in obs:continue
                x=torch.as_tensor(obs[key],dtype=self.cls.dtype,device=self.cls.device).flatten()
                if size is None:
                    pad=(-x.numel())%self.patch
                    x=F.pad(x,(0,pad)).reshape(-1,self.patch)
                else:
                    if x.numel()!=size:raise ValueError(f"Embedding size mismatch: {key}")
                    x=x.reshape(1,size)
                chunks.append(self.projections[key](x)+self.type_embeddings[key])
            tokens=torch.cat(chunks,0)
            if tokens.shape[0]>self.max_tokens:raise ValueError("Too many tokens")
            outputs.append(tokens)
        length=max(len(x) for x in outputs)
        mask=torch.stack([torch.arange(length,device=self.cls.device)>=len(x) for x in outputs])
        batch=torch.stack([F.pad(x,(0,0,0,length-len(x))) for x in outputs])+self.pos[:,:length]
        return self.norm(self.body(batch,src_key_padding_mask=mask)[:,0])

class TorchDecisionHead(nn.Module):
    def __init__(self,width,classes):
        super().__init__()
        self.classes=list(classes)
        self.linear=nn.Linear(width,len(classes))
    def forward(self,features):return self.linear(features)
