# fullgo ensemble LAFA submission container.
# Heavy by design: 6 frozen PLMs are baked in (~20 GB) so query embedding needs
# no network at run time. The CAFA_forever maintainers run this on an A100 with
# the t0 reference bundle bind-mounted at /app/data and queries at /app/input.
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV HF_HOME=/app/.cache/huggingface TRANSFORMERS_CACHE=/app/.cache/huggingface
RUN apt-get update && apt-get install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Bake the 6 PLM weights so the container is offline at run time.
RUN python3 - <<'PY'
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer
for m in ["ElnaggarLab/ankh-base","facebook/esm2_t36_3B_UR50D","ElnaggarLab/ankh-large",
          "facebook/esm2_t33_650M_UR50D"]:
    AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m)
T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
# esmc_600m comes from the `esm` package; pre-download per its API.
PY

COPY predict.py entrypoint.sh ./
# Trained weights (from storage/fullgo_models/): classifier_6plm_asl.pt,
# ensemble_gbm_{NK,LK,PK}.txt, feature_spec.json, v227_exp_freq.tsv
COPY models/ /app/models/
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
