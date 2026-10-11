"""Built-in domain adapters. Preserve scientific metadata outside feature arrays."""
import json
import re
from pathlib import Path
import numpy as np

def _signal(values, n=128):
    x=np.asarray(values,dtype=np.float64).reshape(-1)
    if len(x)<2 or not np.isfinite(x).all(): raise ValueError("Requires >=2 finite samples")
    res=np.interp(np.linspace(0,1,n),np.linspace(0,1,len(x)),x)
    return ((res-res.mean())/(res.std()+1e-6)).astype(np.float32)

class ScientificEncoder:
    """Arrays or CSV/XDI trace samples; no physical-axis assumptions."""
    def __init__(self,size=128): self.size=size
    def __call__(self, observation):
        if isinstance(observation,(list,tuple,np.ndarray)): return _signal(observation,self.size)
        if isinstance(observation,dict):
            if "values" in observation:return _signal(observation["values"],self.size)
            observation=observation.get("path")
        path=Path(observation)
        if path.suffix.lower() not in (".csv",".xdi",".txt",".dat"):
            raise ValueError("Unsupported scientific trace format; use a plugin")
        samples=[]
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith(("#",";")):continue
            parts=re.split(r"[,\s]+",line.strip())
            try: samples.append(float(parts[-1]))
            except (ValueError,IndexError): continue
        return _signal(samples,self.size)

class RasterEncoder:
    """Raster arrays; GeoTIFF via rasterio plugin dependency."""
    def __init__(self,size=128):self.size=size
    def __call__(self,observation):
        if isinstance(observation,(str,Path)):
            import rasterio
            with rasterio.open(observation) as dataset:
                arr=dataset.read(1,masked=True).filled(np.nan)
        else: arr=np.asarray(observation)
        flat=np.asarray(arr,dtype=np.float64).reshape(-1)
        good=flat[np.isfinite(flat)]
        return _signal(good,self.size)

class GeospatialEncoder:
    """Coordinates in supplied CRS; distance is NOT calculated."""
    def __call__(self,observation):
        if not isinstance(observation,dict):raise ValueError("Expected coordinate object")
        lat,lon=observation["lat"],observation["lon"]
        if not (-90<=lat<=90 and -180<=lon<=180):raise ValueError("Invalid WGS84 coordinates")
        return np.array([lat/90,lon/180],dtype=np.float32)

class ImageEncoder:
    """Frozen pretrained CLIP image features, lazily loaded."""
    def __init__(self,model="openai/clip-vit-base-patch32",device="cpu"):
        self.model_name=model;self.device=device;self._model=None
    def __call__(self,observation):
        from PIL import Image
        import torch
        from transformers import CLIPModel,AutoProcessor
        if self._model is None:
            self._model=CLIPModel.from_pretrained(self.model_name).eval().to(self.device)
            self._model.requires_grad_(False)
            self._processor=AutoProcessor.from_pretrained(self.model_name)
        with Image.open(observation) as image:
            inputs=self._processor(images=image.convert("RGB"),return_tensors="pt")
        with torch.inference_mode():
            output=self._model.get_image_features(**{k:v.to(self.device) for k,v in inputs.items()})
            if not isinstance(output,torch.Tensor):output=output.pooler_output
        return output.flatten().float().cpu().numpy()

class VideoEncoder:
    """Bounded frame aggregation; decoder is OpenCV optional."""
    def __init__(self,max_frames=8,image_encoder=None):
        self.max_frames=max_frames;self.image_encoder=image_encoder or ImageEncoder()
    def __call__(self,observation):
        import cv2
        from tempfile import NamedTemporaryFile
        capture=cv2.VideoCapture(str(observation))
        if not capture.isOpened():raise ValueError("Cannot decode video")
        frames=[]
        try:
            count=max(1,int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            sample=set(np.linspace(0,count-1,min(count,self.max_frames),dtype=int).tolist())
            i=0
            while len(frames)<len(sample):
                ok,frame=capture.read()
                if not ok:break
                if i in sample:
                    with NamedTemporaryFile(suffix=".png") as tmp:
                        cv2.imwrite(tmp.name,frame)
                        frames.append(self.image_encoder(tmp.name))
                i+=1
        finally:capture.release()
        if not frames:raise ValueError("Video contained no frames")
        return np.mean(np.stack(frames),axis=0).astype(np.float32)

class AudioEncoder:
    """Frozen Wav2Vec2 mean-pooled features. Requires 16kHz audio."""
    def __init__(self,model="facebook/wav2vec2-base-960h",device="cpu"):
        self.model_name=model;self.device=device;self._model=None
    def __call__(self,observation):
        import soundfile as sf
        import torch
        from transformers import AutoFeatureExtractor,Wav2Vec2Model
        samples,rate=sf.read(observation,dtype="float32",always_2d=True)
        if rate!=16000:raise ValueError("Audio must be 16 kHz; resample explicitly")
        if self._model is None:
            self._model=Wav2Vec2Model.from_pretrained(self.model_name).eval().to(self.device)
            self._model.requires_grad_(False)
            self._extractor=AutoFeatureExtractor.from_pretrained(self.model_name)
        x=self._extractor(samples.mean(axis=1),sampling_rate=16000,return_tensors="pt")
        with torch.inference_mode():
            y=self._model(**{k:v.to(self.device) for k,v in x.items()}).last_hidden_state.mean(1)
        return y[0].float().cpu().numpy()

class LogsEncoder:
    """Safe, simple log/trace statistics; no semantic training required."""
    def __call__(self,observation):
        if isinstance(observation,(str,Path)) and Path(str(observation)).is_file():
            lines=Path(observation).read_text(encoding="utf-8").splitlines()
        else:lines=str(observation).splitlines()
        if not lines:lines=[""]
        joined="\n".join(lines).lower()
        return np.array([np.log1p(len(lines)),np.log1p(len(joined)),joined.count("error")/len(lines),joined.count("warn")/len(lines),joined.count("timeout")/len(lines)],dtype=np.float32)
