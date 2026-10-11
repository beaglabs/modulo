"""Optional Workers AI Clef-Omni teacher; never called implicitly."""
import os
import requests

class ClefTeacher:
    def __init__(self,account_id=None,token=None,session=None):
        self.account_id=account_id or os.getenv("CLOUDFLARE_ACCOUNT_ID")
        self.token=token or os.getenv("CLOUDFLARE_API_TOKEN")
        self.session=session or requests.Session()
    def decide(self,state,questions,*,images=None,audio=None,videos=None):
        if not self.account_id or not self.token:raise ValueError("Cloudflare credentials required")
        body={"state":state,"questions":questions}
        for key,value in (("images",images),("audio",audio),("videos",videos)):
            if value is not None:body[key]=value
        r=self.session.post(f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/@cf/cloudflare/clef-omni",headers={"Authorization":f"Bearer {self.token}"},json=body,timeout=90)
        r.raise_for_status()
        data=r.json()
        if data.get("success") is False:raise RuntimeError(str(data.get("errors")))
        return data.get("result",data)
