import json
from pathlib import Path
from .core import SystemOne

def create_app(config_path):
    from fastapi import FastAPI,HTTPException
    model=SystemOne.from_config(json.loads(Path(config_path).read_text()))
    app=FastAPI(title="Modulo System 1")
    @app.get("/health")
    def health():return {"status":"ok","domain":model.domain}
    @app.post("/v1/decide")
    def decide(observation:dict):
        try:return model.decide(observation).__dict__
        except (ValueError,KeyError,TypeError) as e:raise HTTPException(422,str(e))
    return app
