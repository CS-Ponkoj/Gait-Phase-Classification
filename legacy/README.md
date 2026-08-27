# Legacy Training Workspace

`training_workspace/` is the original `Gait` directory preserved intact.

It contains historical notebooks, scripts, training and test folders, extracted Dataset A content, and four model checkpoints. Keeping the directory intact preserves its internal relative paths and makes later experiment reconstruction possible.

Known limitations include inconsistent class counts, overwritten checkpoint paths in one script, validation-only evaluation, frame-level splitting, missing environment metadata, and metrics that do not establish performance beyond the majority-class baseline.

Do not refactor or delete this evidence workspace until a clean reproducible pipeline has been implemented and validated separately.
