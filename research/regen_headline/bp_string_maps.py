"""Precompute STRING v12.0 network features for mouse (10090) and rat (10116).

For each UniProt accession that maps into STRING for these species:
  degree_clean            = # partners with experimental>0 OR coexpression>0
  degree_hc               = # partners with experimental>=700 OR coexpression>=700
  annotated_partner_count = # of those clean partners that carry a t0 BP annotation
                            (from GAF v225 mouse+rat BP-annotated accessions)

Writes storage/regen_headline/bp_string_features.json : {uniprot_ac: {...}}.
Clean channels only (experimental + coexpression); textmining/database dropped to
avoid literature that could postdate t0.
"""
import gzip, json, collections
from pathlib import Path

STR = Path("/home/frapercan/Thesis2/storage/string_v12")
SC = Path("/tmp/claude-1000/-home-frapercan-Thesis2/afd2c43a-ede7-46dc-94dd-9745808d2194/scratchpad")
OUT = Path("/home/frapercan/Thesis2/storage/regen_headline/bp_string_features.json")
SPECIES = ["10090", "10116"]

# ---- t0 BP-annotated accessions (from GAF extract) ----
bp_annot = set()
gaf = SC / "mouse_rat_bp_annotated.tsv"
if gaf.exists():
    for line in open(gaf):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2:
            bp_annot.add(parts[1])
print(f"[gaf] t0 BP-annotated mouse+rat accessions: {len(bp_annot)}", flush=True)

feat = {}
for sp in SPECIES:
    # aliases: STRING id <-> UniProt AC
    sid2ac = collections.defaultdict(set)
    ac2sid = {}
    with gzip.open(STR / f"tax{sp}.protein.aliases.v12.0.txt.gz", "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            sid, alias, source = line.rstrip("\n").split("\t")
            if source in ("UniProt_AC", "Ensembl_UniProt"):
                sid2ac[sid].add(alias)
                ac2sid.setdefault(alias, sid)
    # which STRING ids carry a t0 BP annotation (any of their UniProt ACs)
    sid_annot = {sid for sid, acs in sid2ac.items() if acs & bp_annot}
    print(f"[{sp}] string ids with UniProt AC: {len(sid2ac)}; annotated: {len(sid_annot)}", flush=True)

    # links.detailed: clean-channel adjacency
    deg = collections.Counter()
    deg_hc = collections.Counter()
    annp = collections.Counter()
    with gzip.open(STR / f"tax{sp}.protein.links.detailed.v12.0.txt.gz", "rt") as fh:
        header = fh.readline()
        for line in fh:
            f = line.split()
            # cols: protein1 protein2 neighborhood fusion cooccurence coexpression experimental database textmining combined
            p1, p2 = f[0], f[1]
            coexp = int(f[5]); exp = int(f[6])
            if exp > 0 or coexp > 0:
                deg[p1] += 1
                if p2 in sid_annot:
                    annp[p1] += 1
                if exp >= 700 or coexp >= 700:
                    deg_hc[p1] += 1
    # attribute to UniProt ACs
    for sid, acs in sid2ac.items():
        d = deg.get(sid, 0)
        for ac in acs:
            # keep the max-degree mapping if AC maps to several sids
            prev = feat.get(ac)
            if prev is None or d > prev["degree_clean"]:
                feat[ac] = {"taxid": sp, "string_id": sid,
                            "degree_clean": d, "degree_hc": deg_hc.get(sid, 0),
                            "annotated_partner_count": annp.get(sid, 0)}

json.dump(feat, open(OUT, "w"))
print(f"[done] wrote {len(feat)} UniProt AC network features -> {OUT}", flush=True)
