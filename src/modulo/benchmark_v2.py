"""Reproducible paired benchmark from curated V2 spectra and frozen image features.

Run: python -m modulo.benchmark_v2 --dataset datasets/usgs-v2 --output results
Image features are deterministic pixel statistics, intentionally a smoke-test
baseline; replace with a frozen pretrained vision encoder for scientific results.
"""
import argparse
import json
import random
from pathlib import Path
import numpy as np
from .spectral_dataset import _read

def features(row, root):
    from PIL import Image
    values=np.asarray(row["reflectance"],dtype=np.float32)
    mask=np.asarray(row["valid_mask"],dtype=bool) & np.isfinite(values)
    if not mask.any(): raise ValueError("no valid spectral values")
    # interpolate missing positions rather than leaking NaNs into the student
    axis=np.arange(len(values))
    spectrum=np.interp(axis,axis[mask],values[mask]).astype(np.float32)
    with Image.open(root/row["photo_path"]) as im:
        pixels=np.asarray(im.convert("RGB").resize((32,32)),dtype=np.float32)/255
    pooled=pixels.reshape(8,4,8,4,3).mean(axis=(1,3)).reshape(-1)
    return np.concatenate([spectrum, mask.astype(np.float32), pooled])

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--epochs",type=int,default=25)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--width",type=int,default=128)
    args=p.parse_args()
    import torch
    from torch.nn import functional as F
    from .masking import MaskedStudent, mask_penalty
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    root=Path(args.dataset)
    partitions={name:_read(root/f"{name}.parquet") for name in ("train","validation","test")}
    labels=sorted({r["mineral"] for rows in partitions.values() for r in rows})
    mapping={s:i for i,s in enumerate(labels)}
    datasets={}
    for name,rows in partitions.items():
        if not rows: raise ValueError(f"empty partition {name}")
        datasets[name]=(torch.tensor(np.stack([features(r,root) for r in rows])),
                        torch.tensor([mapping[r["mineral"]] for r in rows],dtype=torch.long))
    # A genuine benchmark needs class coverage in train AND each held-out partition
    train_classes=set(datasets["train"][1].tolist())
    if any(not set(v[1].tolist())<=train_classes for k,v in datasets.items() if k!="train"):
        raise ValueError("held-out labels absent from training; curate/split classes before benchmarking")
    device="cuda" if torch.cuda.is_available() else "cpu"
    results={}
    for mode in ("none","single","independent","shared"):
        torch.manual_seed(args.seed)
        model=MaskedStudent(datasets["train"][0].shape[1],len(labels),
                             width=args.width,mode=mode).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
        x,y=datasets["train"]
        for epoch in range(args.epochs):
            model.train()
            order=torch.randperm(len(x),generator=torch.Generator().manual_seed(args.seed+epoch))
            for ids in order.split(args.batch_size):
                optimizer.zero_grad()
                logits,masks=model(x[ids].to(device),return_masks=True)
                loss=F.cross_entropy(logits,y[ids].to(device))
                if masks:loss=loss+0.001*mask_penalty(masks)
                loss.backward();optimizer.step()
        model.eval()
        metrics={}
        with torch.no_grad():
            for split in ("validation","test"):
                xx,yy=datasets[split]
                logits,masks=model(xx.to(device),return_masks=True)
                probs=logits.softmax(-1).cpu()
                prediction=probs.argmax(-1)
                accuracy=(prediction==yy).float().mean().item()
                f1s=[]
                for cls in range(len(labels)):
                    tp=((prediction==cls)&(yy==cls)).sum().item()
                    fp=((prediction==cls)&(yy!=cls)).sum().item()
                    fn=((prediction!=cls)&(yy==cls)).sum().item()
                    if (yy==cls).any():
                        f1s.append(2*tp/max(1,2*tp+fp+fn))
                metrics[split]={"accuracy":accuracy,"macro_f1":float(np.mean(f1s)),
                    "cross_entropy":F.cross_entropy(logits,yy.to(device)).item(),
                    "mean_gate":float(mask_penalty(masks).item()) if masks else None}
        results[mode]={"parameters":sum(q.numel() for q in model.parameters()),"metrics":metrics}
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    report={"seed":args.seed,"classes":labels,"dataset_counts":{k:len(v[0]) for k,v in datasets.items()},
            "device":device,"runs":results,
            "limitations":"Image input is pooled RGB, not pretrained image embeddings. A single seed is not a scientific conclusion."}
    (output/"benchmark.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
