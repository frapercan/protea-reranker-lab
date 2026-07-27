"""PHYLO build: map UniProt AC -> OrthoDB gene -> Eukaryota(2759) OG; build OG x species
presence profiles. LEAKAGE-SAFE: uses ONLY genes.tab (UniProt AC + species) and OG2genes
(orthology). gene_xrefs.tab (which carries GO terms) is NEVER read. Orthology/genome content
is annotation-independent and OrthoDB v12.2 (2024) predates t0 (2025-09-04).

Outputs (under storage/phylo_profile/):
  acc2og.json        : accession -> sorted list of at2759 OG ids (mapped via OrthoDB)
  og_members.json    : OG id -> sorted list of accessions among our AC set that belong to it
  og_profiles.npz    : ogs (array of OG ids), taxa (euk ncbi taxid universe),
                       P (scipy CSR presence matrix len(ogs) x len(taxa), binary),
                       plus target_ogs mask
  build_phylo.json   : coverage stats
"""
import gzip, json, time, collections
from pathlib import Path
import numpy as np
from scipy import sparse

t0 = time.time()
P = Path("/home/frapercan/Thesis2/storage/phylo_profile")
GENES = P / "odb12v2_genes.tab.gz"
OG2 = P / "odb12v2_OG2genes.tab.gz"


def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)


# ---- species universe (euk ncbi taxa) + odbsp->tax ----
prep = json.load(open(P / "prep_species.json"))
sp2tax = prep["sp2tax"]                 # odb_species_id -> ncbi taxid
euk_tax = prep["euk_tax"]               # sorted distinct euk ncbi taxids
tax_idx = {t: i for i, t in enumerate(euk_tax)}
euk_sp = set(prep["euk_odbsp"])
log(f"species universe: {len(euk_tax)} euk ncbi taxa from {len(euk_sp)} odb assemblies")

# ---- AC set + which are targets ----
ac_set = set(x.strip() for x in open(P / "ac_set.txt") if x.strip())
import pandas as pd
tg = pd.read_csv("/home/frapercan/Thesis2/storage/regen_headline/cg3_acc_taxon.tsv",
                 sep="\t", header=None, names=["acc", "tax"], dtype=str)
target_acc = set(tg.acc)
log(f"AC set {len(ac_set):,}; targets {len(target_acc):,}")

# ---- pass over genes.tab: UniProt AC (col4) in ac_set -> geneid (col0); geneid prefix -> odbsp ----
ac2gene = collections.defaultdict(set)     # accession -> set(geneid)
gene2acc = {}                              # geneid -> accession (our AC)
n = 0
with gzip.open(GENES, "rt") as fh:
    for ln in fh:
        n += 1
        if n % 20_000_000 == 0:
            log(f"  genes.tab {n:,} lines; matched {len(gene2acc):,}")
        c = ln.rstrip("\n").split("\t")
        if len(c) < 5:
            continue
        uni = c[4]
        if not uni or uni not in ac_set:
            continue
        gid = c[0]
        ac2gene[uni].add(gid)
        gene2acc[gid] = uni
log(f"genes.tab: {n:,} lines; {len(ac2gene):,} accessions mapped to {len(gene2acc):,} genes")

my_genes = set(gene2acc)

# ---- pass1 over OG2genes: at2759 OGs; geneid in my_genes -> OG ----
gene2og = collections.defaultdict(set)     # our geneid -> set(OG at 2759)
n = 0
with gzip.open(OG2, "rt") as fh:
    for ln in fh:
        n += 1
        if n % 40_000_000 == 0:
            log(f"  OG2genes pass1 {n:,} lines")
        tab = ln.find("\t")
        og = ln[:tab]
        if not og.endswith("at2759"):
            continue
        gid = ln[tab + 1:].rstrip("\n")
        if gid in my_genes:
            gene2og[gid].add(og)
log(f"OG2genes pass1: {n:,} lines; {len(gene2og):,} of our genes have an at2759 OG")

# accession -> OGs ; OG -> our accessions
acc2og = collections.defaultdict(set)
og_members = collections.defaultdict(set)
for gid, ogs in gene2og.items():
    acc = gene2acc[gid]
    for og in ogs:
        acc2og[acc].add(og)
        og_members[og].add(acc)

target_ogs = set()
for a in target_acc:
    target_ogs |= acc2og.get(a, set())
# annotated OGs = OGs containing >=1 of our BP-annotated ref accessions (all non-target ACs are BP ref;
# targets may also be annotated but their frozen terms are excluded per-accession at gate time)
relevant_ogs = set(og_members)             # every OG that contains any of our AC-set proteins
log(f"acc mapped: {len(acc2og):,}; target OGs {len(target_ogs):,}; relevant OGs {len(relevant_ogs):,}")

# ---- pass2 over OG2genes: for relevant OGs collect ALL member species (presence profile) ----
og_species = collections.defaultdict(set)  # OG -> set(ncbi taxid) among euk universe
n = 0
with gzip.open(OG2, "rt") as fh:
    for ln in fh:
        n += 1
        if n % 40_000_000 == 0:
            log(f"  OG2genes pass2 {n:,} lines")
        tab = ln.find("\t")
        og = ln[:tab]
        if og not in relevant_ogs:
            continue
        gid = ln[tab + 1:].rstrip("\n")
        odbsp = gid[:gid.find(":")]
        tax = sp2tax.get(odbsp)
        if tax is not None and tax in tax_idx:
            og_species[og].add(tax)
log(f"OG2genes pass2: profiles built for {len(og_species):,} OGs")

# ---- build sparse presence matrix ----
ogs = sorted(relevant_ogs)
og_idx = {o: i for i, o in enumerate(ogs)}
rows, cols = [], []
for og in ogs:
    i = og_idx[og]
    for tax in og_species.get(og, ()):
        rows.append(i); cols.append(tax_idx[tax])
Pmat = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                         shape=(len(ogs), len(euk_tax)))
prev = np.asarray(Pmat.sum(axis=1)).ravel()
tmask = np.array([1 if o in target_ogs else 0 for o in ogs], dtype=np.int8)
log(f"presence matrix {Pmat.shape}; nnz {Pmat.nnz:,}; median OG prevalence {np.median(prev):.0f}")

sparse.save_npz(P / "og_presence.npz", Pmat)
np.savez(P / "og_meta.npz", ogs=np.array(ogs), taxa=np.array(euk_tax),
         prevalence=prev, target_mask=tmask)
json.dump({o: sorted(v) for o, v in acc2og.items()}, open(P / "acc2og.json", "w"))
json.dump({o: sorted(v) for o, v in og_members.items()}, open(P / "og_members.json", "w"))

stats = {
    "orthodb_version": "v12.2 (2024)", "level": "Eukaryota (2759)",
    "euk_taxa_profile_dim": len(euk_tax),
    "ac_set": len(ac_set), "targets": len(target_acc),
    "accessions_mapped_to_orthodb": len(acc2og),
    "targets_mapped": sum(1 for a in target_acc if a in acc2og),
    "target_mapping_frac": round(sum(1 for a in target_acc if a in acc2og) / len(target_acc), 4),
    "relevant_ogs": len(ogs), "target_ogs": len(target_ogs),
    "presence_nnz": int(Pmat.nnz),
    "median_og_prevalence": float(np.median(prev)),
}
json.dump(stats, open(P / "build_phylo.json", "w"), indent=2)
log("wrote build_phylo.json / og_presence.npz / og_meta.npz / acc2og.json / og_members.json")
print(json.dumps(stats, indent=2))
