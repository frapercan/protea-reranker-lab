API Reference
=============

Full autodoc for every public module in :mod:`protea_reranker_lab`,
grouped by the stage of the pipeline it serves. Modules whose source name
starts with an underscore are internal helpers; their public members are
still documented here for completeness. The convenience re-exports on the
top-level package (``from protea_reranker_lab import build_dataset`` and
friends) are documented on their defining module pages below.

Schemas and contracts
---------------------

.. toctree::
   :maxdepth: 1

   schemas
   contracts
   experiment

Dataset building and IO
------------------------

.. toctree::
   :maxdepth: 1

   builder
   data
   bucket_io
   sequences
   multi_source

Staging and splitting
---------------------

.. toctree::
   :maxdepth: 1

   staging
   pooled_staging
   splits
   propagation

Training and inference
----------------------

.. toctree::
   :maxdepth: 1

   reranker
   runner
   train
   universal_runner
   universal_train
   ia_weighting
   k_augmentation
   calibration

Evaluation and analysis
-----------------------

.. toctree::
   :maxdepth: 1

   evaluate
   recall
   compare
   bootstrap
   hierarchical_correction
   negative_sampling_audit
   band_registry_bridge

Dense vs sparse (T-CIENCIA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   sdr
   encoder_ablation
