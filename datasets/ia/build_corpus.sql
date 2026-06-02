-- Export the SwissProt experimental(+IC+TAS) GO annotation corpus for IA.
--
-- Source: PROTEA DB table protein_go_annotation, annotation_set v227
--   (5e84d6c5-a104-407d-8c02-b6b653b571b7), which is sourced from
--   goa_uniprot_all. The join to the protein table (all rows reviewed=true,
--   i.e. SwissProt) plus the verified fact that every experimental accession
--   in v227 maps to a reviewed protein restricts the corpus to SwissProt
--   without an explicit reviewed=true predicate. The predicate is kept here
--   defensively so the query stays correct if unreviewed rows are ever loaded.
--
-- Evidence codes resolve democafa's default 'Experimental,IC,TAS'.
-- NOT-qualified annotations are excluded (matches democafa).
-- Obsolete GO terms are dropped (obonet ignores them anyway).
--
-- Output columns match democafa's expected input: EntryID, term, aspect.
--
-- Usage:
--   psql "postgresql://protea:protea@localhost:5432/protea" \
--     -c "\copy ( <this query> ) TO 'swissprot_exp_v227_raw.tsv' \
--         WITH (FORMAT csv, DELIMITER E'\t', HEADER true)"

SELECT DISTINCT
    pga.protein_accession AS "EntryID",
    g.go_id               AS term,
    g.aspect              AS aspect
FROM protein_go_annotation pga
JOIN go_term g  ON g.id = pga.go_term_id
JOIN protein  p ON p.accession = pga.protein_accession
WHERE pga.annotation_set_id = '5e84d6c5-a104-407d-8c02-b6b653b571b7'
  AND p.reviewed = true
  AND pga.evidence_code IN (
        'EXP','IDA','IPI','IMP','IGI','IEP',
        'HTP','HDA','HMP','HGI','HEP',
        'IC','TAS')
  AND (pga.qualifier IS NULL OR pga.qualifier NOT ILIKE 'NOT%')
  AND g.is_obsolete = false;
