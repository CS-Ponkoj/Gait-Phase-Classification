# Gait Phase Classification

Research project investigating temporal modeling for gait phase classification from silhouette sequences, with an implemented Python codebase for data preparation, training, evaluation, and reproducible analysis.

The accompanying paper, **Temporal Modeling under Deterministic Gait Supervision: A Subject-Independent CASIA C Study**, examines how temporal models reproduce four operational gait states and how the origin of those labels affects the interpretation of model performance. The study connects model comparisons with supervision provenance, subject-independent evaluation, and boundary-event analysis.

## Research overview

Automatically labeling gait sequences makes large-scale model development possible, but agreement with a labeling rule does not establish agreement with physical gait phases. This research studies that distinction using CASIA C thermal silhouettes and deterministic targets derived from silhouette-spread intervals.

The paper addresses three questions:

1. How well do visual models reproduce the operational targets for unseen subjects?
2. What information do neighboring frames and their temporal order contribute?
3. How do target construction and boundary-event counting change the interpretation of the results?

The implementation supports four states, P0–P3, using the vocabulary below. These names describe the intended phase interpretation; the current deterministic labels are provisional and do not independently establish anatomical phase or same-foot continuity.

| State | Operational phase vocabulary |
| --- | --- |
| P0 | Initial contact / loading |
| P1 | Mid-stance |
| P2 | Terminal stance / pre-swing |
| P3 | Swing |

## Method

**Thermal GaitPhaseNet (TGPN)** processes 27-frame clips of aligned, single-channel silhouettes. Its implemented pipeline combines:

- An EfficientNet-B0 encoder with global and lower-half pooling of a shared feature map.
- An adjacent-frame difference branch for motion features.
- Residual temporal convolutions with dilations 1, 2, 4, and 8, followed by self-attention.
- Dense four-state predictions and a boundary prediction for each frame.
- State and boundary losses, with temporal smoothing and cyclic-order regularization.

Probabilities are averaged across overlapping clips during inference. The model uses both past and future frames within a clip; the study does not establish causal or real-time operation.

Comparisons include a majority baseline, HOG-style SVM, a single-frame CNN, and an ordered temporal convolutional network (TCN). Repeated-frame and shuffled-context TCN controls examine the contribution of temporal inputs. A cycle-position diagnostic is reported separately because it receives privileged target-generation metadata rather than visual input alone.

Implementation details and training settings are documented in the [TGPN guide](docs/THERMAL_GAITPHASENET.md), [model implementation](src/gait_phase/models.py), and [study configuration](configs/study.yaml).

## Evaluation approach

The study compares temporal and single-frame models under subject-independent evaluation, with controlled experiments examining neighboring-frame information and temporal order. Evaluation considers subject-level classification, boundary timing, and the relationship between supervision provenance and model predictions.

The codebase supports development folds, held-out evaluation safeguards, subject-level metrics, bootstrap uncertainty estimates, and generation of manuscript tables and figures from saved predictions. The [development run registry](configs/development_runs_v1.yaml) records the experiment organization. Trained weights, predictions, and generated evidence remain local and are not included in a standard clone.

## Data and evaluation scope

The paper focuses on **CASIA C thermal silhouette sequences** across normal, slow, fast, and bag-carrying walking conditions. The repository also preserves an earlier CASIA A curated subset and historical single-frame experiments. Those materials provide research history and are separate from the paper's four-state CASIA C analysis. OU-ISIR is excluded from this study.

Subject partitions separate development and held-out evaluation. Development launchers do not open the test partition. Final neural evaluation requires explicit test acknowledgement and a fixed training duration selected from development results; test data is not used for early stopping or checkpoint selection.

Independent anatomical annotation and expert adjudication remain necessary for evaluating physical gait phases. The research scope is operational gait-state modeling; clinical applications and cross-dataset generalization require separate validation.

Dataset access must be obtained under the source datasets' terms. Raw images, prepared image copies, and generated checkpoints are excluded from normal Git tracking. See the [data statement](docs/DATA_STATEMENT.md) and [annotation protocol](docs/annotation_protocol.md).

## Reproducing the implementation

### Environment

Use **Python 3.11** from the repository root. The following PowerShell commands create an environment, install the pinned dependencies, and install the package:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m gait_phase.cli --help
```

GPU configuration is described in the [reproduction guide](docs/reproducibility.md). Training launchers use the repository environment. Pretrained encoder weights may be downloaded on first use.

### Prepare licensed data

Place the required source data at the paths defined in [configs/study.yaml](configs/study.yaml), then run:

```powershell
.\.venv\Scripts\python.exe -m gait_phase.cli extract-casia-c
.\.venv\Scripts\python.exe -m gait_phase.cli build-manifest
.\.venv\Scripts\python.exe -m gait_phase.cli validate-data data/manifests/generated/casia_a_c_manifest.csv
.\.venv\Scripts\python.exe -m gait_phase.cli make-splits data/manifests/generated/casia_a_c_manifest.csv --output data/manifests/generated/casia_a_c_split.csv
.\.venv\Scripts\python.exe -m gait_phase.cli auto-annotate data/manifests/generated/casia_a_c_split.csv
.\.venv\Scripts\python.exe -m gait_phase.cli prepare-training-data data/manifests/generated/casia_a_c_split.csv
```

Preparation creates the local provisional release and verifies copied-image checksums and fold manifests before writing `READY.json`. A source-code clone alone does not include the datasets or prepared training trees.

### Train and analyze development experiments

Run a one-epoch pipeline check on fold 0:

```powershell
.\scripts\train-provisional.ps1 -Model tgpn -Folds 0 -Epochs 1 -BatchSize 2 -Device auto
```

This checks the training workflow; its score is not a paper result. For the five-fold experiment:

```powershell
.\scripts\train-provisional.ps1 -Model tgpn -Folds 0,1,2,3,4 -Epochs 30 -BatchSize 2 -Device auto
```

Use the [TGPN guide](docs/THERMAL_GAITPHASENET.md) for baseline comparisons, temporal controls, and final evaluation safeguards. Each run records its configuration, environment, checkpoint, predictions, metrics, and learning curves where applicable under `artifacts/runs/`.

Aggregate the registered development runs when their local artifacts are available:

```powershell
.\.venv\Scripts\python.exe -m gait_phase.cli aggregate-results --registry configs/development_runs_v1.yaml --workspace . --output artifacts/paper_assets/development_v2 --bootstrap-iterations 2000
```

The registry references specific saved runs; newly generated runs must be registered before aggregating them. Analysis can also be generated from an individual run's saved predictions using the `evaluate` and `make-paper-assets` commands. See the [reproduction guide](docs/reproducibility.md) for details.

### Validate the codebase

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Tests cover configuration, manifests, annotations, splitting, prepared data, model contracts, metrics, experiment summaries, and the command-line workflow. Running them does not reproduce GPU training or validate anatomical labels.

## Repository organization

| Path | Purpose |
| --- | --- |
| [src/gait_phase/](src/gait_phase/) | Research pipeline, models, training, evaluation, and paper-asset generation |
| [configs/](configs/) | Study settings and registered development experiments |
| [scripts/](scripts/) | Training launchers, environment setup, and annotation audit utilities |
| [tests/](tests/) | Automated implementation checks |
| [docs/](docs/) | Method, annotation, data, and reproduction documentation |
| [docs/paper/](docs/paper/) | Earlier manuscript and paper outline |
| [annotations/](annotations/) | Annotation protocols and local release organization |
| [data/](data/) | Local source data, manifests, and prepared datasets |
| [artifacts/](artifacts/) | Local experiment outputs and generated evidence |
| [inventory/](inventory/) | Preserved data-lineage and inventory records |
| [legacy/](legacy/) | Original single-frame research code and historical experiments |

The earlier manuscript in `docs/paper/` describes the initial ResNet152V2/InceptionV3 study. The research framing here follows the revised temporal-modeling manuscript. Some dated documentation describes earlier stages of the work; consult the current code, study configuration, and run registry for implementation details.
