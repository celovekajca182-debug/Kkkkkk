import numpy as np

class TinyCharRNN:
    """Учебная символьная RNN: предсказывает следующий символ."""
    def __init__(self, chars, hidden=96, seed=7):
        self.chars=list(chars); self.stoi={c:i for i,c in enumerate(self.chars)}
        self.itos={i:c for c,i in self.stoi.items()}; self.vocab=len(self.chars); self.hidden=hidden
        rng=np.random.default_rng(seed); s=0.08
        self.Wxh=rng.normal(0,s,(hidden,self.vocab)).astype(np.float32)
        self.Whh=rng.normal(0,s,(hidden,hidden)).astype(np.float32)
        self.Why=rng.normal(0,s,(self.vocab,hidden)).astype(np.float32)
        self.bh=np.zeros((hidden,1),np.float32); self.by=np.zeros((self.vocab,1),np.float32)

    def train_text(self,text,epochs=20,seq_len=48,lr=0.06):
        ids=[self.stoi[c] for c in text if c in self.stoi]
        if len(ids)<seq_len+2: raise ValueError("Мало данных: добавь более длинный корпус.")
        for ep in range(epochs):
            hprev=np.zeros((self.hidden,1),np.float32); total=0; batches=0
            for p in range(0,len(ids)-seq_len-1,seq_len):
                xs=ids[p:p+seq_len]; ys=ids[p+1:p+seq_len+1]
                hs={-1:hprev}; probs={}; loss=0.0
                for t,x in enumerate(xs):
                    xv=np.zeros((self.vocab,1),np.float32); xv[x]=1
                    hs[t]=np.tanh(self.Wxh@xv+self.Whh@hs[t-1]+self.bh)
                    z=self.Why@hs[t]+self.by; z-=z.max()
                    q=np.exp(z); q/=q.sum(); probs[t]=q
                    loss-=float(np.log(q[ys[t],0]+1e-12))
                total+=loss; batches+=1
                dWxh=np.zeros_like(self.Wxh); dWhh=np.zeros_like(self.Whh)
                dWhy=np.zeros_like(self.Why); dbh=np.zeros_like(self.bh); dby=np.zeros_like(self.by)
                dhnext=np.zeros_like(hprev)
                for t in reversed(range(len(xs))):
                    dy=probs[t].copy(); dy[ys[t]]-=1
                    dWhy+=dy@hs[t].T; dby+=dy
                    dh=self.Why.T@dy+dhnext; raw=(1-hs[t]*hs[t])*dh
                    dbh+=raw; xv=np.zeros((self.vocab,1),np.float32); xv[xs[t]]=1
                    dWxh+=raw@xv.T; dWhh+=raw@hs[t-1].T; dhnext=self.Whh.T@raw
                for g in (dWxh,dWhh,dWhy,dbh,dby): np.clip(g,-5,5,out=g)
                self.Wxh-=lr*dWxh; self.Whh-=lr*dWhh; self.Why-=lr*dWhy
                self.bh-=lr*dbh; self.by-=lr*dby; hprev=hs[len(xs)-1]
            print(f"epoch {ep+1}/{epochs} loss={total/max(1,batches):.3f}")

    def generate(self,prompt="",length=180,temperature=.8):
        h=np.zeros((self.hidden,1),np.float32); out=prompt
        seed=prompt or self.chars[0]
        for ch in seed:
            x=np.zeros((self.vocab,1),np.float32); x[self.stoi.get(ch,0)]=1
            h=np.tanh(self.Wxh@x+self.Whh@h+self.bh)
        idx=self.stoi.get(seed[-1],0); rng=np.random.default_rng()
        for _ in range(length):
            x=np.zeros((self.vocab,1),np.float32); x[idx]=1
            h=np.tanh(self.Wxh@x+self.Whh@h+self.bh)
            z=(self.Why@h+self.by)/max(.05,temperature); z-=z.max()
            p=np.exp(z); p/=p.sum()
            idx=int(rng.choice(self.vocab,p=p[:,0])); out+=self.itos[idx]
        return out

    def save(self,path):
        np.savez_compressed(path,chars=np.array(self.chars),hidden=self.hidden,
            Wxh=self.Wxh,Whh=self.Whh,Why=self.Why,bh=self.bh,by=self.by)

    @classmethod
    def load(cls,path):
        d=np.load(path,allow_pickle=False); obj=cls(d["chars"].tolist(),int(d["hidden"]))
        for k in ("Wxh","Whh","Why","bh","by"): setattr(obj,k,d[k].astype(np.float32))
        return obj
