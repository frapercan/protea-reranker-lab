"""Extracting per-residue, per-layer representations for the bounded probe.

One forward pass per window, overlaps merged, every residue kept, float32
throughout. This is the only place in the mechanism study that needs the card, and
it exists so the residue tensor has to be built once rather than at corpus scale.

FOUR THINGS THAT ARE CORRECTNESS RATHER THAN STYLE

* **float32 end to end.** Middle layers carry massive activations, layer 38 of
  ankh-base measured at 440,611 against float16's 65,504. Running the model in
  half precision overflows them DURING the forward pass, not merely on the way to
  disk, so the dtype is a property of the extraction and not of the storage. It
  cost a purity of 0.076 once, which is the value random noise gives.
* **Ankh's tokenisation is not the obvious one.** Its SentencePiece tokeniser maps
  a literal space to unknown, so a space-joined sequence silently becomes a string
  of unknowns. It takes a list of characters with ``is_split_into_words=True``,
  and non-standard residues are mapped to X first. Mirrored from the production
  backend so the probe measures the representation production serves.
* **Only the trailing EOS is stripped.** Ankh adds no CLS, so residue index i is
  amino acid i once that one token is dropped. Getting this off by one shifts
  every residue against its position and nothing downstream can see it.
* **The alignment is asserted, not assumed.** A window's extracted residue count
  must equal the window's length. This is the same shape as the join that once
  rewrote 8.3 per cent of rows: silent, plausible, and wrong by one position.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np

from protea_reranker_lab.residue_probe import ProbeSpec, chunk_spans

log = logging.getLogger(__name__)

#: Non-standard residues Ankh's vocabulary does not carry, mapped as production does.
NONSTANDARD = re.compile(r"[UZOB]")


@dataclass(frozen=True)
class ExtractionResult:
    """Residues for one protein: (length, layers, width), float32."""

    accession: str
    residues: np.ndarray
    windows: int

    @property
    def length(self) -> int:
        return int(self.residues.shape[0])


def load_encoder(model_name: str = "ElnaggarLab/ankh-base", device: str = "cuda"):
    """Tokeniser and encoder in float32, mirroring the production loader.

    float32 even on the card, deliberately: half precision is what production uses
    for pooled inference and is what corrupts mid-layer extraction, and this
    function exists to extract mid layers.
    """
    import torch
    from transformers import AutoTokenizer, T5EncoderModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name, torch_dtype=torch.float32)
    model = model.to(torch.device(device)).eval()
    log.info("loaded %s on %s in float32", model_name, device)
    return tokenizer, model


def _window_layers(tokenizer, model, window: str, layers: tuple[int, ...]) -> np.ndarray:
    """One window's hidden states, (window_length, layers, width), float32.

    Raises when the extracted residue count disagrees with the window, because a
    silent off-by-one here shifts every residue against its amino acid and every
    number computed afterwards is about a sequence nobody wrote.
    """
    import torch

    cleaned = NONSTANDARD.sub("X", window)
    inputs = tokenizer.batch_encode_plus(
        [list(cleaned)],
        padding="longest",
        truncation=False,
        add_special_tokens=True,
        is_split_into_words=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True)
    states = out.hidden_states
    del out

    # No CLS in Ankh, so only the trailing EOS comes off.
    keep = int(inputs["attention_mask"][0].sum().item()) - 1
    if keep != len(window):
        raise ValueError(
            f"window of {len(window)} residues produced {keep} positions. The "
            "tokenisation is not one token per residue, so every residue would be "
            "attributed to the wrong position"
        )
    stacked = torch.stack([states[i][0, :keep, :] for i in layers], dim=1)
    return stacked.float().cpu().numpy()


def extract_protein(tokenizer, model, sequence: str, spec: ProbeSpec) -> ExtractionResult:
    """Every residue of one protein, windows merged by averaging their overlap.

    Averaging rather than taking the last window: a residue near a boundary has
    seen less context in one window than in the other, and preferring either is a
    choice nobody would be able to see afterwards. The mean is not neutral either,
    but it is symmetric and it is stated.
    """
    spans = chunk_spans(len(sequence), spec.chunk_size, spec.chunk_overlap)
    total = np.zeros((len(sequence), len(spec.layers), spec.width), dtype=np.float64)
    counts = np.zeros(len(sequence), dtype=np.int32)

    for start, end in spans:
        block = _window_layers(tokenizer, model, sequence[start:end], spec.layers)
        total[start:end] += block
        counts[start:end] += 1

    if int(counts.min()) == 0:
        raise ValueError("some residues were covered by no window; the spans do not tile")
    merged = (total / counts[:, None, None]).astype(np.float32)
    if not np.isfinite(merged).all():
        raise ValueError(
            "extracted residues contain non-finite values. Middle layers carry "
            "activations far above the float16 ceiling, so this is what half "
            "precision looks like after the fact"
        )
    return ExtractionResult(accession="", residues=merged, windows=len(spans))


def extract_probe(sequences: dict[str, str], spec: ProbeSpec, *,
                  model_name: str = "ElnaggarLab/ankh-base",
                  device: str = "cuda") -> dict[str, ExtractionResult]:
    """The whole probe, in accession order so a rerun writes the same artifact."""
    tokenizer, model = load_encoder(model_name, device)
    out: dict[str, ExtractionResult] = {}
    for i, accession in enumerate(sorted(sequences), start=1):
        result = extract_protein(tokenizer, model, sequences[accession], spec)
        out[accession] = ExtractionResult(accession, result.residues, result.windows)
        if i % 20 == 0 or i == len(sequences):
            log.info("extracted %d/%d proteins", i, len(sequences))
    return out


def save_probe(path, extracted: dict[str, ExtractionResult], spec: ProbeSpec) -> None:
    """One archive: a concatenated residue matrix plus the offsets that split it.

    Concatenated rather than one array per protein, because an npz with hundreds of
    members is slow to open and hides the total from anyone inspecting it.
    """
    accessions = sorted(extracted)
    lengths = np.array([extracted[a].length for a in accessions], dtype=np.int64)
    matrix = np.concatenate([extracted[a].residues for a in accessions], axis=0)
    np.savez_compressed(
        path,
        accessions=np.array(accessions, dtype=object),
        lengths=lengths,
        offsets=np.concatenate([[0], np.cumsum(lengths)]),
        residues=matrix,
        layers=np.array(spec.layers, dtype=np.int64),
        windows=np.array([extracted[a].windows for a in accessions], dtype=np.int64),
    )
    log.info("wrote %d proteins, %d residues, %.2f GB",
             len(accessions), int(lengths.sum()), matrix.nbytes / 1024 ** 3)


def load_probe(path) -> dict[str, np.ndarray]:
    """Accession to (length, layers, width), split back out of the archive."""
    with np.load(path, allow_pickle=True) as archive:
        accessions = [str(a) for a in archive["accessions"]]
        offsets = archive["offsets"]
        residues = archive["residues"]
        if int(offsets[-1]) != residues.shape[0]:
            raise ValueError(
                f"offsets end at {int(offsets[-1])} and the matrix holds "
                f"{residues.shape[0]} rows; the archive is inconsistent and every "
                "protein after the first mismatch would read another's residues"
            )
        return {
            a: residues[int(offsets[i]):int(offsets[i + 1])]
            for i, a in enumerate(accessions)
        }
