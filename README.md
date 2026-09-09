
# Boundary-Specific Uncertainty in Transformer-Based Named Entity Recognition

## From Boundary-Aware to Uncertainty-Adaptive Conformal Scoring

This repository contains the **final consolidated Kaggle notebooks** used in the experimental pipeline for the MSc dissertation:

> **Boundary-Specific Uncertainty in Transformer-Based Named Entity Recognition: From Boundary-Aware to Uncertainty-Adaptive Conformal Scoring**

The study investigates whether **start- and end-boundary uncertainty can be quantified separately in Named Entity Recognition (NER)** and whether incorporating those signals into **conformal prediction** improves coverage-efficiency behaviour.

Two Transformer-based NER architectures are studied:

1. **BERT BIO sequence labelling**
2. **Custom BERT-based SpanNER-style span classification**

---

## Experimental Outputs

Due to their size, the full experimental outputs and frozen artefacts are not stored directly in this GitHub repository.

The complete experimental artefact package corresponding to the final dissertation notebooks is hosted on Kaggle:

**https://www.kaggle.com/datasets/umrahzargar/20793907-final-experimental-artefacts/data**

The Kaggle package contains the saved outputs required by the consolidated experimental pipeline, including model outputs, prediction artefacts, calibration outputs, conformal prediction artefacts, robustness results, locked CoNLL-2003 test outputs, and WNUT-17 replication outputs.

> If the Kaggle dataset is currently private, the link will only be accessible to authorised users until the dataset is made public.

---

## Research Pipeline

```text
Data & Split Registry
        ↓
BERT Baseline & Uncertainty
        ↓
SpanNER Baseline & Uncertainty
        ↓
Architecture & Uncertainty Comparison
        ↓
Probability & Boundary Calibration
        ↓
Conformal Prediction: M1–M4
        ↓
Matched-Width & M4 Mechanism Analysis
        ↓
Controlled Noise Robustness
        ↓
Final Locked CoNLL-2003 Test
        ↓
WNUT-17 External Replication
````

---

## Repository Structure

| Notebook | Description                           |
| -------- | ------------------------------------- |
| `00`     | Data, EDA & Split Registry            |
| `01`     | BERT Baseline & Uncertainty           |
| `02`     | SpanNER Baseline & Uncertainty        |
| `03`     | Architecture & Uncertainty Comparison |
| `04`     | Probability & Boundary Calibration    |
| `05`     | Conformal Prediction & M1–M4          |
| `06`     | Matched Width & M4 Mechanism Analysis |
| `07`     | Controlled Noise Robustness           |
| `08`     | Final Locked CoNLL Test               |
| `09`     | WNUT-17 External Replication          |

Recommended execution order:

```text
00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09
```

---

# Key Verified Results

The final consolidated notebook package was re-audited notebook-by-notebook to ensure that downstream analyses consume the intended **frozen upstream artefacts**. 

## CoNLL-2003 SpanNER Validation Baseline

| Metric    |        Value |
| --------- | -----------: |
| TP        |        2,866 |
| FP        |          112 |
| FN        |          145 |
| Precision |     0.962391 |
| Recall    |     0.951843 |
| F1        | **0.957088** |

## SpanNER Boundary-Error Discrimination

| Boundary |    AUROC |    AUPRC |
| -------- | -------: | -------: |
| Start    | 0.852633 | 0.048918 |
| End      | 0.848764 | 0.039329 |

## Temperature Scaling

| Architecture | Temperature |
| ------------ | ----------: |
| BERT         |    1.264385 |
| SpanNER      |    1.466519 |

## Frozen CoNLL Lambda Configurations

| Architecture | Method | λ Start | λ End |
| ------------ | ------ | ------: | ----: |
| BERT         | M2     |    0.01 |  0.01 |
| BERT         | M3     |    0.00 |  0.01 |
| BERT         | M4     |    0.00 |  0.01 |
| SpanNER      | M2     |    1.00 |  1.00 |
| SpanNER      | M3     |    1.00 |  1.00 |
| SpanNER      | M4     |    1.00 |  0.30 |

For M2, the start and end values represent the same shared λ.

## SpanNER 90% Conformal Quantiles

| Method |       q̂ |
| ------ | -------: |
| M1     | 0.149860 |
| M2     | 0.177472 |
| M3     | 0.177472 |
| M4     | 0.155677 |

## Final Held-Out CoNLL-2003 NER Performance

| Architecture | Strict Typed-Span F1 |
| ------------ | -------------------: |
| BERT         |             0.910240 |
| SpanNER      |         **0.915846** |

## Held-Out CoNLL-2003 Conformal Results at 90% Target Coverage

| Architecture | Method |     Coverage | Mean Set Size |
| ------------ | ------ | -----------: | ------------: |
| BERT         | M1     |     0.847379 |      2.191138 |
| BERT         | M2     |     0.847089 |      2.190849 |
| BERT         | M3     |     0.847089 |      2.190849 |
| BERT         | M4     |     0.847089 |      2.190849 |
| SpanNER      | M1     |     0.845641 |      1.507674 |
| SpanNER      | M2     | **0.850854** |      1.513756 |
| SpanNER      | M3     | **0.850854** |      1.513756 |
| SpanNER      | M4     |     0.846510 |      1.507964 |

## Controlled-Noise SpanNER Performance

| Condition      | Strict F1 |
| -------------- | --------: |
| Clean          |  0.957088 |
| 5% corruption  |  0.909422 |
| 10% corruption |  0.869307 |
| 20% corruption |  0.806194 |

## WNUT-17 External Replication

### Native strict F1

| Architecture |           F1 |
| ------------ | -----------: |
| BERT         |     0.388659 |
| SpanNER      | **0.431421** |

### Width ≤ 4 strict F1

| Architecture |           F1 |
| ------------ | -----------: |
| BERT         |     0.395909 |
| SpanNER      | **0.439644** |

---

# Notebook Guide

## 00 Data, EDA & Split Registry

### Purpose

Notebook 00 defines the authoritative dataset populations used throughout the CoNLL-2003 pipeline.

It:

* loads the CoNLL-2003 data;
* performs the final exploratory and integrity checks;
* retains the official **14,041-sentence training population**;
* deterministically partitions the original 3,250-sentence development population;
* saves split indices, mappings, metadata and registry artefacts;
* reserves the official CoNLL-2003 test set for the final locked evaluation.

### Frozen Study Populations

| Population              | Sentences |
| ----------------------- | --------: |
| Training                |    14,041 |
| Validation              |     1,625 |
| Probability calibration |       975 |
| Conformal calibration   |       650 |

**Random seed:** `42`

Notebook 00 is the authoritative source for split membership. Downstream notebooks reconstruct their populations from the saved registry rather than creating new random splits. 

---

## 01 BERT Baseline & Uncertainty

### Purpose

Notebook 01 implements the BERT BIO branch of the study.

It:

* fine-tunes `bert-base-cased` for BIO token classification;
* applies first-subword label alignment;
* reconstructs typed entity spans from deterministic BIO predictions;
* evaluates strict span-level NER performance;
* constructs the project-specific geometric typed-span confidence measure;
* extracts BERT-specific start- and end-boundary uncertainty signals;
* performs strict span matching and boundary-error analysis;
* runs **10-pass MC Dropout** for stochastic uncertainty analysis;
* attaches stochastic uncertainty features to the deterministic prediction population;
* saves models, predictions, matching tables, uncertainty features and evaluation outputs.

> **Important:** MC Dropout is used to derive uncertainty features and analyse predictive instability. It does **not** redefine the underlying deterministic BERT predictions.

The official CoNLL-2003 test split is not evaluated in this notebook. 

---

## 02 SpanNER Baseline & Uncertainty

### Purpose

Notebook 02 implements the custom BERT-based SpanNER-style branch.

It:

* uses `bert-base-cased` as the contextual encoder;
* enumerates candidate spans up to a maximum width of **4 tokens**;
* classifies each candidate as an entity type or `NONE`;
* applies deterministic non-overlapping span decoding;
* evaluates strict span-level NER performance;
* saves candidate-level logits, probabilities and decoded predictions;
* constructs architecture-specific start- and end-boundary competition signals;
* creates correctness and boundary-error event structures equivalent to the BERT branch;
* runs 10-pass MC Dropout without changing deterministic candidate or decoded-span populations;
* saves candidate-, span-, boundary- and stochastic-uncertainty artefacts.

### Verified Validation Result

```text
TP        = 2866
FP        = 112
FN        = 145
Precision = 0.962391
Recall    = 0.951843
F1        = 0.957088
```

The final SpanNER model bundle is provenance-checked downstream using its saved **SHA-256 fingerprint**.

Probability calibration and conformal prediction are not fitted in this notebook.

The official CoNLL-2003 test split is not evaluated in this notebook. 

---

## 03 Architecture & Uncertainty Comparison

### Purpose

Notebook 03 places BERT and SpanNER into a common validation-stage comparison.

It:

* compares strict span-level performance;
* performs the matched-width control required by SpanNER's width ≤ 4 restriction;
* compares architecture-native uncertainty signals;
* evaluates boundary-error discrimination using **AUROC** and **AUPRC**;
* evaluates selective prediction using **risk-coverage analysis**;
* produces figures, tables and frozen analysis artefacts;
* verifies the attached SpanNER snapshot before analysis.

### Verified SpanNER Boundary Results

```text
Start boundary:
AUROC = 0.852633
AUPRC = 0.048918

End boundary:
AUROC = 0.848764
AUPRC = 0.039329
```

This remains a development/validation-stage analysis. No calibration or conformal procedures are fitted here. 

---

## 04 Probability & Boundary Calibration

### Purpose

Notebook 04 performs probability and correctness calibration while preserving the frozen architecture outputs.

It:

* verifies the resolved BERT and SpanNER artefact chains;
* fits one multiclass temperature parameter per architecture using only the probability-calibration population;
* propagates temperature-scaled probabilities into downstream candidate-, span- and boundary-level quantities;
* fits separate binary calibrators for:

  * exact-entity correctness,
  * start-boundary correctness,
  * end-boundary correctness,
  * joint-boundary correctness;
* evaluates calibration using:

  * Negative Log-Likelihood,
  * Brier score,
  * Expected Calibration Error;
* uses independent validation to determine whether binary mappings improve probabilistic calibration;
* freezes calibration decisions before conformal prediction.

### Verified Temperatures

```text
BERT    = 1.264385
SpanNER = 1.466519
```

### Calibration Questions

This stage separates two questions:

1. Can each architecture's probability outputs be improved through temperature scaling?
2. Can entity and boundary correctness probabilities be calibrated separately?

Temperature scaling is propagated into the primary **M1–M4 conformal experiment**.

The binary correctness calibrators are retained as secondary diagnostics and do **not** replace the deterministic architecture-specific boundary-uncertainty signals used by M4. 

---

## 05 Conformal Prediction & M1–M4

### Purpose

Notebook 05 contains the primary conformal prediction experiment.

It:

* constructs architecture-native typed-span candidate universes;
* implements conformal scoring methods M1–M4;
* uses the fixed validation population only for λ selection;
* performs grouped five-fold pseudo-conformal cross-fitting;
* freezes scoring definitions and λ values before conformal calibration;
* uses the reserved 650-sentence conformal-calibration population to estimate finite-sample conformal thresholds;
* evaluates coverage and prediction-set efficiency;
* saves the frozen configurations and conformal artefacts used downstream;
* validates the upstream model and calibration provenance before proceeding.

### Conformal Methods

| Method | Description                                                                          |
| ------ | ------------------------------------------------------------------------------------ |
| **M1** | Temperature-scaled confidence-only baseline                                          |
| **M2** | Shared fixed start/end boundary-rank penalty                                         |
| **M3** | Separate fixed start/end boundary-rank penalties                                     |
| **M4** | Uncertainty-adaptive rank penalties using architecture-specific boundary uncertainty |

For M4:

* **BERT** uses BIO-derived boundary uncertainty.
* **SpanNER** uses candidate-local boundary competition.

Binary boundary-correctness calibrators and MC-Dropout uncertainty are **not** inputs to the primary M1–M4 scores.

### Frozen Lambda Values

```text
BERT
M2: shared λ = 0.01
M3: λ_start = 0.00, λ_end = 0.01
M4: λ_start = 0.00, λ_end = 0.01

SpanNER
M2: shared λ = 1.00
M3: λ_start = 1.00, λ_end = 1.00
M4: λ_start = 1.00, λ_end = 0.30
```

### SpanNER 90% q̂ Values

```text
M1 = 0.149860
M2 = 0.177472
M3 = 0.177472
M4 = 0.155677
```

The reserved conformal-calibration population is accessed only after λ selection and scoring definitions have been frozen.

The CoNLL-2003 test set is not used for method selection or conformal calibration. 

---

## 06 Matched Width & M4 Mechanism Analysis

### Purpose

Notebook 06 provides deeper diagnostic analysis of the frozen conformal experiment.

It:

* re-examines BERT and SpanNER under matched candidate-width conditions;
* investigates why M4 behaves differently from fixed M2/M3 alternatives;
* analyses candidate-score changes, rank penalties and uncertainty-dependent relaxation;
* studies method-specific conformal quantiles;
* investigates SpanNER candidate-local ambiguity;
* retains selected M5/confidence-gating experiments strictly as post-hoc diagnostics;
* verifies that Notebook 05 outputs originate from the same frozen artefact root.

No downstream λ retuning is permitted.

This notebook is **interpretive and diagnostic**, not a new method-selection stage. 

---

## 07 Controlled Noise Robustness

### Purpose

Notebook 07 evaluates behaviour under controlled synthetic text corruption.

It:

* introduces character-level noise at multiple severities;
* reruns model inference under corrupted inputs;
* measures strict NER degradation;
* examines changes in boundary uncertainty with increasing corruption;
* evaluates conformal behaviour using the already-frozen clean-data methodology;
* produces the robustness figures and tables reported in the dissertation;
* verifies the resolved SpanNER model and Notebook 05 conformal handoff.

### Verified SpanNER Strict F1

```text
Clean          = 0.957088
5% corruption  = 0.909422
10% corruption = 0.869307
20% corruption = 0.806194
```

This stage investigates whether boundary-specific uncertainty behaves meaningfully as input quality deteriorates and whether clean-calibrated conformal behaviour transfers under controlled distribution shift. 

---

## 08 Final Locked CoNLL Test

### Purpose

Notebook 08 performs the formal held-out evaluation on the official CoNLL-2003 test split.

It:

* verifies the final SpanNER model fingerprint;
* verifies frozen temperatures, λ values and conformal quantiles;
* evaluates the frozen BERT and SpanNER pipelines;
* applies calibration and conformal procedures without test-set retuning;
* reports strict typed-span performance;
* reports held-out conformal coverage and efficiency;
* includes clearly labelled post-hoc diagnostics for interpreting calibration-to-test and nonconformity shifts;
* prevents post-hoc diagnostics from modifying the primary methodology.

### Strict Typed-Span Results

```text
BERT F1    = 0.910240
SpanNER F1 = 0.915846
```

### 90% Conformal Results

```text
BERT
M1: coverage = 0.847379, mean set size = 2.191138
M2: coverage = 0.847089, mean set size = 2.190849
M3: coverage = 0.847089, mean set size = 2.190849
M4: coverage = 0.847089, mean set size = 2.190849

SpanNER
M1: coverage = 0.845641, mean set size = 1.507674
M2: coverage = 0.850854, mean set size = 1.513756
M3: coverage = 0.850854, mean set size = 1.513756
M4: coverage = 0.846510, mean set size = 1.507964
```

Notebook 08 is the **formal locked-test stage**.

Test results are not used to retune:

* models;
* temperature parameters;
* λ values;
* conformal thresholds;
* primary scoring definitions. 

---

## 09 WNUT-17 External Replication

### Purpose

Notebook 09 performs an external replication and sensitivity analysis on WNUT-17.

It follows the same overall methodological protocol while fitting dataset-specific models and calibration quantities.

It:

* trains WNUT-specific BERT and SpanNER models;
* defines WNUT-specific development, probability-calibration and conformal-calibration populations;
* fits WNUT-specific temperatures;
* selects WNUT-specific λ values using the same pre-specified protocol;
* estimates WNUT-specific conformal quantiles;
* evaluates native and matched-width strict NER performance;
* examines conformal coverage and efficiency under dataset/domain shift;
* tests whether uncertainty-adaptive conformal scoring generalises beyond CoNLL-2003.

> **Important:** Notebook 09 does not consume the CoNLL SpanNER, calibration or conformal artefacts from Notebooks 02, 04 or 05. It **replicates the protocol** rather than transferring CoNLL-fitted parameters.

### Verified Native Strict F1

```text
BERT    = 0.388659
SpanNER = 0.431421
```

### Verified Width ≤ 4 Strict F1

```text
BERT    = 0.395909
SpanNER = 0.439644
```

WNUT-17 is treated as an **external sensitivity/replication setting**, not as part of CoNLL method development. 

---

# Dependency Structure

The main CoNLL dependency chain is:

```text
00
├── 01 BERT
├── 02 SpanNER
│
├── 03 Architecture Comparison
│
├── 04 Calibration
│
├── 05 Conformal Prediction
│   ├── 06 Mechanism Analysis
│   └── 07 Controlled Noise
│
└── 08 Final Locked Test

09 WNUT-17 External Replication
   └── Independent dataset-specific replication of the protocol
```

More specifically:

* **Notebook 00** defines the authoritative CoNLL dataset and split registry.
* **Notebooks 01 and 02** depend on Notebook 00.
* **Notebook 03** consumes frozen validation outputs from 01 and 02.
* **Notebook 04** consumes frozen architecture outputs and the probability-calibration population.
* **Notebook 05** consumes architecture outputs, temperature calibration and the predefined validation/conformal-calibration populations.
* **Notebook 06** consumes frozen conformal and architecture-level artefacts.
* **Notebook 07** consumes frozen models and the frozen Notebook 05 conformal specification.
* **Notebook 08** consumes the final frozen CoNLL model, calibration and conformal artefacts.
* **Notebook 09** independently replicates the protocol on WNUT-17.

> For dissertation integrity, Notebook 08 must not be used for development or parameter selection. If it is re-executed for reproducibility, all upstream model, calibration and conformal choices must remain frozen. 

---

# Runtime and Environment

The project was developed and executed primarily using **Kaggle notebooks** and Python.

## Recommended Runtime

* Kaggle Python environment
* GPU accelerator for model training, inference and MC Dropout
* Internet access where Hugging Face resources must be downloaded

## Core Libraries

* Python 3
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* NumPy
* pandas
* scikit-learn
* Matplotlib

Each notebook contains its own:

* imports;
* configuration;
* path resolution;
* integrity checks;
* artefact validation.

Where an upstream Kaggle output dataset is required, the consuming notebook explicitly locates and verifies the expected artefacts before proceeding.

Exact package versions are not hard-coded here because the executed notebooks and Kaggle runtime provide the authoritative environment record. 

---

# Reproducibility and Experimental Discipline

The final notebook sequence follows the following experimental controls:

* Random seed `42` defines the frozen CoNLL development-data partition.
* Dataset populations are reconstructed from saved source indices rather than resampled downstream.
* Validation, probability-calibration and conformal-calibration populations have separate experimental roles.
* The official CoNLL-2003 test split is excluded from training, calibration, λ selection and q̂ estimation.
* Deterministic model predictions remain fixed when MC-Dropout features are calculated.
* SpanNER uses a maximum candidate span width of four tokens.
* Matched-width controls are included when direct architectural comparison requires them.
* Architecture-native uncertainty measures are evaluated through common downstream metrics rather than assuming raw uncertainty values are directly comparable.
* Probability calibration, boundary calibration and conformal scoring are fitted only using their designated non-test populations.
* The held-out CoNLL-2003 test set is used only for final locked evaluation.
* Test inspection does not trigger retrospective retuning of the primary methodology.
* WNUT-17 is treated as a separate external replication with dataset-specific models, calibration and conformal fitting.
* Downstream provenance checks validate critical model fingerprints, temperatures, λ values and conformal quantiles before later stages execute.

The split registry is **index-disjoint**.

Naturally occurring duplicate sentence or token content may still exist across different source rows in the original corpus. The final pipeline preserves the frozen registry rather than redefining partitions post hoc. 

---

# Output Artefacts

The full output package is hosted on Kaggle rather than directly in this repository because of its size.

Typical artefacts produced across the notebook sequence include:

* split/configuration JSON files;
* label mappings;
* model checkpoints and bundles;
* candidate-level prediction tables;
* span-level prediction tables;
* gold-span tables;
* prediction/gold matching and error tables;
* deterministic boundary-uncertainty features;
* MC-Dropout uncertainty features;
* calibration models and calibrated probabilities;
* frozen temperature specifications;
* frozen λ specifications;
* conformal quantiles and component tables;
* conformal prediction outputs;
* comparison tables and figures;
* robustness outputs;
* locked-test outputs;
* external replication outputs;
* protocol/completion metadata.

Later CoNLL notebooks are designed to consume saved upstream artefacts rather than silently recomputing or redefining earlier experimental populations. 

---

# Pipeline at a Glance

| Stage     | Role                                                      |
| --------- | --------------------------------------------------------- |
| **00**    | Define data and frozen experimental populations           |
| **01–02** | Build architecture-specific NER and uncertainty pipelines |
| **03**    | Compare architecture and uncertainty behaviour            |
| **04**    | Calibrate confidence and boundary correctness             |
| **05**    | Freeze and evaluate conformal methods M1–M4               |
| **06**    | Analyse the mechanism behind conformal findings           |
| **07**    | Test robustness under controlled text corruption          |
| **08**    | Perform final locked CoNLL-2003 evaluation                |
| **09**    | Replicate the protocol on WNUT-17                         |

---

# Notes for Reviewers

These notebooks are **final consolidated submission notebooks**, not a complete archive of every exploratory notebook produced during development.

The consolidation preserves:

* the scientifically used methodology;
* the final audited artefact chain;
* the implementation supporting the dissertation;
* the numerical results reported from that chain.

It removes material that is not part of the final project record, including:

* duplicated setup cells;
* obsolete historical experiments;
* debugging material;
* redundant exploratory analysis;
* superseded intermediate branches.

Integrity and provenance audit cells are intentionally retained where they verify critical upstream artefacts.

The dissertation report should be treated as the primary source for:

* research motivation;
* methodological justification;
* equations and notation;
* interpretation of findings;
* limitations;
* discussion;
* conclusions.

This repository provides the **executable experimental implementation**, while the linked Kaggle dataset contains the associated **large experimental artefacts**. 



with your actual Kaggle link, and delete the `` bits.
