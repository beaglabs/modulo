import argparse,json
from pathlib import Path
from .registry import defaults,DOMAINS
from .core import SystemOne
from .training import read_rows,train_head,save_head

def main(argv=None):
    p=argparse.ArgumentParser(prog="modulo")
    s=p.add_subparsers(dest="action",required=True)
    s.add_parser("domains")
    r=s.add_parser("components");r.add_argument("kind",choices=["encoders","adapters","heads"])
    a=s.add_parser("encode");a.add_argument("--domain",choices=DOMAINS,required=True);a.add_argument("--input",required=True);a.add_argument("--output",required=True)
    t=s.add_parser("train");t.add_argument("--input",required=True);t.add_argument("--output",required=True);t.add_argument("--epochs",type=int,default=100)
    d=s.add_parser("decide");d.add_argument("--config",required=True);d.add_argument("--input",required=True)
    v=s.add_parser("serve");v.add_argument("--config",required=True);v.add_argument("--host",default="127.0.0.1");v.add_argument("--port",type=int,default=8000)
    dataset=s.add_parser("dataset")
    ds=dataset.add_subparsers(dest="dataset_action",required=True)
    f=ds.add_parser("fetch");f.add_argument("source",choices=["usgs-v7","usgs-v2"]);f.add_argument("--output");f.add_argument("--url");f.add_argument("--sha256")
    q=ds.add_parser("prepare");q.add_argument("source",choices=["usgs-v7","usgs-v2"]);q.add_argument("--spectral-dataset",default="datasets/usgs-v7");q.add_argument("--crosswalk");q.add_argument("--images-root");q.add_argument("--archive",default="datasets/raw/ASCIIdata_splib07a.zip");q.add_argument("--output");q.add_argument("--wavelength-min",type=float,default=400);q.add_argument("--wavelength-max",type=float,default=2500);q.add_argument("--step-nm",type=float,default=5);q.add_argument("--min-coverage",type=float,default=.6)
    x=ds.add_parser("split");x.add_argument("source",choices=["usgs-v7","usgs-v2"]);x.add_argument("--dataset");x.add_argument("--group-by",choices=["specimen"],default="specimen");x.add_argument("--train",type=float,default=.7);x.add_argument("--val",type=float,default=.15);x.add_argument("--test",type=float,default=.15);x.add_argument("--seed",type=int,default=42)
    y=ds.add_parser("validate");y.add_argument("source",choices=["usgs-v7","usgs-v2"]);y.add_argument("--dataset")
    au=ds.add_parser("audit");au.add_argument("source",choices=["usgs-v2"]);au.add_argument("--dataset",default="datasets/usgs-v2")
    discover=ds.add_parser("discover");discover.add_argument("source",choices=["usgs-v2"]);discover.add_argument("--dataset",default="datasets/usgs-v7");discover.add_argument("--sources",default="datasets/raw/usgs-v2/sources");discover.add_argument("--output",default="datasets/usgs-v2/photo-candidates.csv");discover.add_argument("--download",action="store_true")
    e=ds.add_parser("export");e.add_argument("source",choices=["usgs-v7","usgs-v2"]);e.add_argument("--dataset");e.add_argument("--format",choices=["parquet"],default="parquet");e.add_argument("--output",required=True)
    args=p.parse_args(argv)
    if args.action=="dataset":
        from . import spectral_dataset as sd
        action=args.dataset_action
        from . import dataset_v2 as v2
        if action=="discover":
            from . import photo_discovery
            if args.download: photo_discovery.download_sources(Path(args.sources).parent)
            result=photo_discovery.discover_candidates(args.dataset,args.sources,args.output)
        elif action=="audit": result=v2.audit(args.dataset)
        elif action=="fetch" and args.source=="usgs-v2": result=v2.fetch(args.output or "datasets/raw/usgs-v2",url=args.url,expected_sha256=args.sha256)
        elif action=="fetch": result={"archive":str(sd.fetch(args.output or "datasets/raw/ASCIIdata_splib07a.zip",url=args.url,expected_sha256=args.sha256))}
        elif action=="prepare" and args.source=="usgs-v2":
            result=v2.prepare(args.spectral_dataset,args.crosswalk,args.images_root,args.output or "datasets/usgs-v2",archive="datasets/raw/usgs-v2/ASCIIdata_splib07a.zip")
        elif action=="prepare": result=sd.prepare(args.archive,args.output or "datasets/usgs-v7",start_nm=args.wavelength_min,stop_nm=args.wavelength_max,step_nm=args.step_nm,min_coverage=args.min_coverage)
        elif action=="split" and args.source=="usgs-v2": result=v2.split(args.dataset or "datasets/usgs-v2",seed=args.seed,train=args.train,val=args.val,test=args.test)
        elif action=="split": result=sd.split(args.dataset or "datasets/usgs-v7",seed=args.seed,train=args.train,val=args.val,test=args.test)
        elif action=="validate": result=v2.audit(args.dataset or "datasets/usgs-v2") if args.source=="usgs-v2" else sd.validate(args.dataset or "datasets/usgs-v7")
        elif action=="export" and args.source=="usgs-v2":
            result=v2.export(args.dataset or "datasets/usgs-v2",args.output)
        else: result={"output":str(sd.export(args.dataset or "datasets/usgs-v7",args.output))}
        print(json.dumps(result,indent=2))
    elif args.action=="domains":print("\n".join(DOMAINS))
    elif args.action=="components":print("\n".join(defaults().names(args.kind)))
    elif args.action=="encode":
        data=json.loads(Path(args.input).read_text())
        values=defaults().create("encoders",args.domain)(data).tolist()
        Path(args.output).write_text(json.dumps({"features":values}))
    elif args.action=="train":
        save_head(args.output,train_head(read_rows(args.input),epochs=args.epochs));print(args.output)
    elif args.action=="decide":
        model=SystemOne.from_config(json.loads(Path(args.config).read_text()))
        print(json.dumps(model.decide(json.loads(Path(args.input).read_text())).__dict__))
    elif args.action=="serve":
        from .server import create_app
        import uvicorn
        uvicorn.run(create_app(args.config),host=args.host,port=args.port)

if __name__=="__main__":main()
