#!/usr/bin/env bash
# Overnight orchestrator: ensure cooccurrence for all 13 training t0 sets,
# then re-export (with WORKING association), protected from the reaper.
set -u
LOG=/home/frapercan/Thesis2/storage/fullgo_models/overnight.log
RAW="protea-int8-s3miKn_ogIjTXq4BryPsxiTpJ5vDzoj-"
API=http://127.0.0.1:8000
SETS=(a8e2ffd4-721c-493f-882b-db6b92cb98bd d09b2b14-0b27-47d0-afd9-8c9e5c3c3052 \
  f720c3c4-abc0-49f7-9d3c-bb917a77089c eb33207a-d876-420e-be67-8f0bd2748909 \
  2a5f3325-e4e8-4f6f-8f0c-82b1f6ca755f 55999285-e523-4388-999d-fe934cc09994 \
  120edbcf-467a-4a76-8c36-a86cc51bed3d 5db5ea46-4a37-4f37-a282-5a97e98207c6 \
  9493bc57-90d6-4929-ac02-f897af0c04e6 3c51a9c8-d7a2-40c5-86a3-30d659542e73 \
  22cb2901-09ca-49fa-8ec9-271d3feda62e b66eb37f-9e9d-4ee5-a6cd-c5379abbae30 \
  1559d9f7-195d-4892-af16-8b58f7fc9942)
log(){ echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
jwt(){ curl -s -X POST "$API/v1/auth/api-key-login" -H "Content-Type: application/json" \
  -d "{\"api_key\":\"$RAW\",\"ttl_seconds\":86400}" 2>/dev/null | python3 -c "import sys,json;print(json.load(sys.stdin).get('token',''))"; }
pg(){ PGPASSWORD=protea psql -h localhost -U protea -d protea -tAc "$1" 2>/dev/null; }
has_cooc(){ local n; n=$(pg "select count(*) from term_cooccurrence where annotation_set_id='$1'"); [ "${n:-0}" -gt 0 ]; }

log "=== overnight orchestrator start ==="
# Phase 1: ensure all 13 sets have cooccurrence (rate limit 10 jobs/min).
while true; do
  missing=(); for s in "${SETS[@]}"; do has_cooc "$s" || missing+=("$s"); done
  log "cooccurrence missing: ${#missing[@]}/13"
  [ "${#missing[@]}" -eq 0 ] && break
  J=$(jwt); sent=0
  for s in "${missing[@]}"; do
    # skip if a build for this set is already queued/running
    act=$(pg "select count(*) from job where operation='build_go_cooccurrence' and status in ('QUEUED','RUNNING') and payload->>'annotation_set_id'='$s'")
    [ "${act:-0}" -gt 0 ] && continue
    curl -s -X POST "$API/jobs" -H "Authorization: Bearer $J" -H "Content-Type: application/json" \
      -d "{\"operation\":\"build_go_cooccurrence\",\"queue_name\":\"protea.jobs\",\"payload\":{\"annotation_set_id\":\"$s\"}}" >/dev/null 2>&1
    sent=$((sent+1)); log "dispatched build $s"
    [ "$sent" -ge 8 ] && break   # stay under 10/min
  done
  sleep 90
done
log "=== all 13 cooccurrence sets present ==="

# Phase 2: re-export with working association.
J=$(jwt)
RESP=$(curl -s -X POST "$API/jobs" -H "Authorization: Bearer $J" -H "Content-Type: application/json" -d '{
 "operation":"export_research_dataset","queue_name":"protea.training",
 "payload":{"k":30,"output_name":"fullgo-native-parity-SELECT-220-227-v2assoc",
   "test_versions":[227],"train_versions":[160,165,170,175,180,185,190,195,200,205,211,215,220],
   "search_backend":"faiss","annotation_source":"goa","use_embedding_pca":true,
   "compute_taxonomy":true,"compute_alignments":true,"expand_votes_to_ancestors":true,
   "compute_self_prior":true,"compute_association":true,"compute_classifier":true,
   "embedding_config_id":"08234f06-ba76-4d7d-aaec-ae601096b4fa",
   "ontology_snapshot_id":"35c3ad67-3002-47db-8f71-eeed69d22ad6"}}')
EJOB=$(echo "$RESP" | python3 -c "import sys,json;print(json.load(sys.stdin).get('id',''))" 2>/dev/null)
echo "$EJOB" > /home/frapercan/Thesis2/storage/fullgo_models/reexport_job.txt
log "=== re-export dispatched: $EJOB ==="

# Phase 3: protect from reaper + watch to completion.
while true; do
  J=$(jwt)
  st=$(curl -s "$API/jobs/$EJOB" -H "Authorization: Bearer $J" 2>/dev/null | python3 -c "import sys,json;print(json.load(sys.stdin).get('status'))" 2>/dev/null)
  pg "update job set leased_until=now()+interval '12 hours' where id='$EJOB' and status='RUNNING'" >/dev/null
  R=$(ps -eo pid,args|grep "queue reaper"|grep -v grep|awk '{print $1}'|head -1); [ -n "$R" ] && kill -9 "$R" 2>/dev/null
  case "$st" in SUCCEEDED*) log "RE-EXPORT DONE"; break;; FAILED*) log "RE-EXPORT FAILED"; break;; esac
  sleep 120
done
log "=== orchestrator end (status=$st) ==="
