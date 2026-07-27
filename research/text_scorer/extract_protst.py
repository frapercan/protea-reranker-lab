"""Extract ProtST-ESM1b text-aligned protein_feature (512-d) for the substrate."""
import json, os, numpy as np, torch
from transformers import AutoModel, AutoTokenizer
SUB="/home/frapercan/Thesis2/storage/layer_ablation/seqs.json"
OUT="/home/frapercan/Thesis2/storage/text_scorer"; MAXLEN=1022; BATCH=12
seqs=json.load(open(SUB)); accs=sorted(seqs)
print(f"proteins: {len(accs)}", flush=True)
tok=AutoTokenizer.from_pretrained("facebook/esm1b_t33_650M_UR50S")
model=AutoModel.from_pretrained("mila-intel/ProtST-esm1b", trust_remote_code=True).eval().to("cuda")
bank=np.zeros((len(accs),512),np.float32)
for i in range(0,len(accs),BATCH):
    chunk=accs[i:i+BATCH]
    enc=tok([seqs[a][:MAXLEN] for a in chunk], return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN+2).to("cuda")
    with torch.no_grad():
        out=model.protein_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    feat=out.protein_feature if hasattr(out,"protein_feature") else out["protein_feature"]
    bank[i:i+len(chunk)]=feat.float().cpu().numpy()
    if (i//BATCH)%80==0: print(f"  {i}/{len(accs)}",flush=True)
np.save(os.path.join(OUT,"query_protst.npy"), bank)
json.dump({"accs":accs,"dim":512}, open(os.path.join(OUT,"meta.json"),"w"))
print("DONE ProtST query embeddings saved", flush=True)
