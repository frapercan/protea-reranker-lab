import sys, json, time
sys.path.insert(0, "/home/frapercan/Thesis2/storage/transfew_calib")
import frame
LAB = "/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier/percut_rerank/predictions"
out = {}
for cat in ("lk", "pk"):
    t0 = time.time()
    auth = frame.score_dir(f"{LAB}/{cat}", cat, n_cpu=8)
    # self-check per-protein machinery on the SAME deployed file
    M = frame.build_bp_matrices(f"{LAB}/{cat}/{cat}.tsv", cat, n_cpu=8)
    f_self, tau_self = frame.best_tau_micro_f(M)
    out[cat] = {"cafaeval_BP": auth, "selfcheck_bootstrapengine": {"f": round(f_self,5), "tau": tau_self},
                "n_proteins": int(len(M["prot_ids"])), "secs": round(time.time()-t0,1)}
    print(cat, "cafaeval", round(auth["f_micro_w"],5), "tau", auth["tau"],
          "| engine", round(f_self,5), "tau", tau_self, "| P", len(M["prot_ids"]), flush=True)
json.dump(out, open("anchor_check.json","w"), indent=2)
print("DONE")
