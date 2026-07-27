"""Track B step 2: backfill the six typed LAFA columns from go_prediction.features.

Key-filtered (only rows that carry at least one of the six keys, ~16M of 52M) and
batched by id range so no single transaction is huge and no lock is held long. A
disk self-guard aborts before any batch if free space drops under 20 GB. Autovacuum
reclaims dead tuples between batches; an explicit VACUUM ANALYZE runs at the end.
Read-safe for the live API: row-level locks only, on the batch's rows.

Idempotent: re-running re-sets the same values, so a resume after an abort is safe.
"""
from __future__ import annotations

import subprocess
import time

import psycopg2

DSN = "host=localhost dbname=protea user=protea password=protea"
KEYS = ["classifier_score", "classifier_present", "self_prior_score",
        "association_total", "association_cross", "association_present"]
WIDTH = 500_000
MIN_ID = 133_300_716          # resume point after backfill died mid-run at id 133.3M (7,662,928 rows already done)
MAX_ID = 144_747_268          # exclusive upper (max(id) was 144,747,267)
MIN_FREE_GB = 20


def df_free_gb() -> int:
    out = subprocess.check_output(["df", "--output=avail", "-BG", "/"]).decode().splitlines()[1]
    return int(out.strip().rstrip("G"))


def main() -> None:
    setclause = ", ".join(f"{k} = (features->>'{k}')::float8" for k in KEYS)
    arr = "array[" + ",".join(f"'{k}'" for k in KEYS) + "]"
    sql = (f"UPDATE go_prediction SET {setclause} "
           f"WHERE id >= %s AND id < %s AND features ?| {arr}")

    conn = psycopg2.connect(DSN)
    conn.autocommit = True
    cur = conn.cursor()

    total = 0
    lo = MIN_ID
    t_start = time.time()
    while lo < MAX_ID:
        hi = min(lo + WIDTH, MAX_ID)
        free = df_free_gb()
        if free < MIN_FREE_GB:
            print(f"ABORT: disk free {free} GB < {MIN_FREE_GB} GB at id {lo}; "
                  f"VACUUM then resume from this id.", flush=True)
            cur.execute("VACUUM (ANALYZE) go_prediction")
            break
        t0 = time.time()
        cur.execute(sql, (lo, hi))
        n = cur.rowcount
        total += n
        print(f"[{lo}..{hi}) updated {n:>7} | total {total:>9} | free {free} GB | "
              f"{time.time() - t0:5.1f}s", flush=True)
        lo = hi
        time.sleep(3)   # gentle: yield I/O to the live API between batches (non-blocking backfill)
    else:
        print(f"ALL BATCHES DONE, total updated {total} in {time.time() - t_start:.0f}s. "
              f"VACUUM ANALYZE...", flush=True)
        cur.execute("VACUUM (ANALYZE) go_prediction")
        print("VACUUM ANALYZE done.", flush=True)

    # verification: non-null counts per column + a value spot-check
    for k in KEYS:
        cur.execute(f"SELECT count({k}) FROM go_prediction")
        print(f"  non-null {k}: {cur.fetchone()[0]:,}", flush=True)
    cur.execute("""SELECT id, classifier_score, (features->>'classifier_score')::float8
                   FROM go_prediction
                   WHERE features ? 'classifier_score' AND classifier_score IS NOT NULL
                   LIMIT 3""")
    for row in cur.fetchall():
        ok = abs((row[1] or 0) - (row[2] or 0)) < 1e-9
        print(f"  spotcheck id={row[0]} col={row[1]} jsonb={row[2]} match={ok}", flush=True)
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
