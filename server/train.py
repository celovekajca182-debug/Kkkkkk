import argparse
from model import TinyCharRNN

p=argparse.ArgumentParser()
p.add_argument("--data",default="data/train.txt")
p.add_argument("--epochs",type=int,default=30)
p.add_argument("--hidden",type=int,default=96)
p.add_argument("--output",default="model.npz")
a=p.parse_args()
with open(a.data,encoding="utf-8") as f: corpus=f.read()
chars=sorted(set(corpus))
if len(corpus)<1000: raise SystemExit("Добавь в train.txt хотя бы 1000 символов разнообразного текста.")
model=TinyCharRNN(chars,a.hidden)
model.train_text(corpus,a.epochs)
model.save(a.output)
print("Saved",a.output)
