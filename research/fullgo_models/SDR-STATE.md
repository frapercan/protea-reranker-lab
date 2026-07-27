# Programa SDR: estado completo (2026-06-23)

Síntesis de todo lo investigado sobre representaciones dispersas (sparse.pdf) para transferencia
de función proteica (GO). Todos los números son Spearman intrínseco vs GO-semántica salvo donde se
diga f_micro. Pool v227 t0 leakage-clean; held-out para los arms aprendidos; ankh-base salvo nota.

## 1. La pregunta y las métricas

- **Pregunta:** ¿una representación DISPERSA captura la función (GO) mejor que la DENSA?
- **Métrica intrínseca (cribado barato):** Spearman entre similitud-de-representación y similitud-GO
  (Resnik/Lin) sobre ~200k pares. Mide la GEOMETRÍA. No predice.
- **Métrica extrínseca (producto):** f_micro del k-NN GO-transfer (la tarea real, lo que se paga).
- Distinción clave: Spearman alto != f_micro alto (la tarea usa solo los top-K vecinos + transfiere
  términos ponderados por IC). Spearman cría hipótesis barato; f_micro decide producto.

## 2. La tabla maestra (Resnik intrínseco salvo nota)

| representación | Resnik | qué prueba |
|---|---|---|
| **dense cosine** | **0.22** | baseline |
| sparse-REAL (top-k + magnitudes, coseno) | ~0.22 | **H1: sparsificar es GRATIS** (-0.01) |
| raw magnitude-BINARY (k-WTA naive, Tanimoto) | 0.09 | **H2: binarizar MATA** (-0.13) |
| chunk-SDR naive (sparsify-then-bundle, per-chunk) | 0.13 | **H3 naive: el ORDEN no cambia nada** (= pool-then-sparsify, ambos fallan) |
| GO-prototype aligned-binary (SUPERVISADO, techo) | 0.25 | **H4: un binario ALINEADO bate al denso** (techo "se puede") |
| **SDR-C A0 learned-binary (mean-pooled)** | **0.49** | **H4/H6: un binario APRENDIDO generaliza y bate al denso (+0.27), held-out** |
| A1 learned-binary (per-chunk bundle) | 0.48 | **H3/H6 learned: no-colapsar NO ayuda** (~= A0, leve por debajo) |
| task-aware (IC-weighted + hard-neg) | corriendo | objetivo alineado a Fmax vs geometría |

Extrínseco (f_micro, k-NN GO-transfer, Rung 1):

| | f_micro | índice |
|---|---|---|
| dense | 0.656 | 100% |
| sparse-real k=256 | 0.655 | 33% (3x menor) |
| sparse-real k=128 | 0.649 | 17% (6x menor) |

## 3. La cadena de hallazgos (en orden lógico)

**(a) SDR naive falla, y NO por el pooling.** SDR-A (ProtT5 mean, k-WTA): coseno 0.315 vs Tanimoto
0.255. Re-test justo (ankh per-chunk, sparsify-then-bundle): identico a pool-then-sparsify, ambos
~0.13 < denso 0.22. El orden del bundling no es el lever. Window-agnostico (v227 == v230).

**(b) La descomposición (el experimento clave).** Aislando las 3 variables (dispersar / binarizar /
metrica) sobre el mismo vector: **dispersar cuesta -0.01 (gratis); binarizar cuesta -0.13 (el
killer).** Toda la perdida dense-vs-sparse era TIRAR LAS MAGNITUDES, no quedarse con pocas dims. En
un espacio crudo, la senal vive en las magnitudes; el conjunto de bits activos no significa nada.

**(c) La escalera de validacion de palanca:**
- **Rung 1 (eficiencia): CONFIRMADA.** sparse-real ~= dense en la TAREA (f_micro), indice 6x menor
  por -0.007. Es una palanca de producto shippable (indice mas pequeno/rapido, misma senal).
- **Rung 2 (precision alcanzable): CONFIRMADA (techo).** Un binario ALINEADO con funcion (prototipos
  GO, supervisado) recupera el -0.13 y bate al denso (0.25 vs 0.22). En un espacio alineado,
  binarizar deja de doler (la senal se muda al PATRON DE BITS). Prueba que el binario PUEDE codificar
  funcion; es supervisado (techo), no free lunch.
- **Rung 3 (SDR-C learned): EL GRAN POSITIVO.** Un encoder lineal aprendido (768->2048, top-k,
  contrastivo soft-Tanimoto sobre GO-sim) da, en proteinas HELD-OUT, **learned-binary = 0.49 Resnik /
  0.50 Lin vs denso 0.22.** Generaliza (proteinas no vistas), bate al denso por +0.27, y supera el
  techo supervisado de Rung 2. La curva subio monotona 150 epochs sin overfit. ES el sueno SDR
  funcionando: un codigo binario disperso APRENDIDO > denso.
  - Matiz: si entrenas el objetivo COSENO-real (no el binario), el real va a 0.60 pero binariza mal
    (0.13) -> hay que optimizar el objetivo BINARIO (Tanimoto sobre el conjunto activo).
  - Bug aprendido: ReLU -> dead units / colapso a density 0; encoder lineal lo arregla.

**(d) El grid (granularidad x metodo x tipo).** A0=mean, A1=per-chunk, A2=per-residuo (pendiente).
- **A1 (per-chunk learned bundle) = 0.48 ~= A0 (0.49).** No-colapsar (sparsify-then-bundle) NO bate
  al mean cuando aprendes. El encoder se recupera bien del mean-pooling; mantener los chunks anade
  ruido. (Responde la preocupacion "estamos colapsando con mean": el colapso NO era el problema.)
- **Estratificacion por tamano (H5):** A1 va PEOR en grandes/multi-chunk (0.45) que en pequenas
  (0.49) -> las grandes son mas DIFICILES, no mas faciles, para el per-chunk. Contraintuitivo.
  PENDIENTE: A0 estratificado para el test limpio "per-chunk vs mean en grandes".

## 4. Conclusiones (que esta cerrado)

- **CONFIRMADO H1:** sparsificar es gratis (real-valued sparse ~= denso).
- **CONFIRMADO H2:** binarizar es el coste (-0.13), no la dispersion.
- **NEGATIVO H3:** el orden sparsify-then-bundle vs pool-then-sparsify no cambia nada (naive); y per-
  chunk learned (A1) ~= mean learned (A0). No-colapsar no paga.
- **CONFIRMADO H4:** un binario ALINEADO con funcion bate al denso (supervisado 0.25; aprendido 0.49).
- **CONTRA H5 (provisional):** las proteinas grandes van PEOR con per-chunk, no mejor. Falta A0-strat.
- **CONFIRMADO H6 (parcial):** learned x binario (A0) es el ganador (0.49). La variante no-colapso
  (A1) no mejora. El cuadrante per-residuo (A2) sigue pendiente.

**La gran historia:** "no es sparse pierde, es BINARIZAR pierde; y binarizar deja de doler cuando
APRENDES un codigo alineado con funcion". Un SDR binario aprendido bate al denso y generaliza. La
palanca de eficiencia (sparse-real) ya es shippable; la de precision (learned binary) esta validada
intrinsecamente y pendiente de confirmar en f_micro.

## 5. Pendiente

- **task-aware (corriendo):** objetivo IC-weighted + hard-negatives -> sube el f_micro vs el de
  geometria pura? El puente ciencia(Spearman)->producto(Fmax).
- **A0 estratificado:** para cerrar H5 (per-chunk vs mean en grandes).
- **A2 per-residuo:** re-embed per-residuo de la muestra (GPU; arreglado, esperando liberar GPU) ->
  sparsify-each-residue-then-bundle (naive + learned). El sparse.pdf section 2 literal.
- **Producto:** entrenar SDR-C a escala (pool completo) + validar en f_micro (227-230) + integrar
  como branch metric="tanimoto" + EvidenceScorer (ADR-D43). Numpy/FAISS, NUNCA pgvector.

## 6. Artefactos

Scripts en `protea-reranker-lab/scripts/`: run_sdr_chunk_correlation.py (decomposicion+ladder),
run_knn_transfer_sparse.py (Rung 1 f_micro), run_sdr_aligned_binary.py (Rung 2 prototipos),
run_sdr_c_contrastive.py (SDR-C A0), run_sdr_c_chunk.py (A1), run_sdr_c_taskaware.py (task-aware).
Extraccion per-residuo: `storage/fullgo_models/extract_per_residue.py`. MLflow: experimentos
sdr-chunk-correlation, sdr-knn-transfer, sdr-aligned-binary, sdr-c-binary-objective, sdr-c-chunk-bundle,
sdr-c-taskaware. Plan: `agent-farm/plans/SDR-GRID.md`, `SDR-LEVER-VALIDATION.md`.
