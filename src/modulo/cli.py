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
    args=p.parse_args(argv)
    if args.action=="domains":print("\n".join(DOMAINS))
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
