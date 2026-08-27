# Manifest Schema

The generated manifest is the authoritative link between licensed images, annotations, subject-independent partitions, and saved predictions. Raw images remain local and ignored by Git.

| Field | Meaning |
|---|---|
| `sample_id` | Stable identifier derived from dataset and relative source path |
| `dataset` | `casia_a_curated` or `casia_c` |
| `subject_id` | Dataset-namespaced subject identifier |
| `sequence_id` | Dataset-namespaced uninterrupted sequence identifier |
| `condition` | Dataset condition, such as `fn00`, `fs00`, `fq00`, or `fb00` |
| `view` | Recorded or documented camera view |
| `frame_index` | Original frame number |
| `cycle_id` | Adjudicated gait-cycle identifier; blank before annotation |
| `cycle_position` | Normalized position in the adjudicated cycle, in `[0,1)` |
| `relative_path` | Workspace-relative local image path |
| `sha256` | Exact content checksum |
| `perceptual_hash` | Difference hash used to flag visually similar candidates |
| `duplicate_group` | Exact-duplicate group identifier, if applicable |
| `legacy_label` | Historical three-class label retained only for provenance |
| `annotator_1_label`, `annotator_2_label` | Independently derived four-phase labels |
| `adjudicated_label` | Final four-phase reference label |
| `annotation_confidence` | `high`, `medium`, or `low` |
| `exclusion_reason` | Reason the sample is excluded from primary analysis |
| `split` | `development` or locked `test` |
| `fold` | Development fold `0`-`4`; test uses `-1` |

Allowed final labels are locked in this order:

1. `initial_contact_loading`
2. `mid_stance`
3. `terminal_stance_preswing`
4. `swing`

The frozen-manifest validation gate rejects blank eligible labels, missing splits, invalid labels, subject leakage, cross-partition exact duplicates, and identical content with conflicting adjudicated labels.
