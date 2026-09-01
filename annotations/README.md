# Annotations

- `templates/` contains tracked schemas and examples.
- `working/` contains local independent annotations and is ignored by Git.
- `adjudicated/` contains local expert decisions and is ignored until a release review confirms consent, licensing, and de-identification requirements.
- `pilot/` contains blinded local image bundles, previews, coordinator keys, and independent annotation files; it is ignored by Git.
- `provisional/` contains versioned machine-generated annotation metadata. It never represents human or clinical ground truth.
- `releases/` is reserved for independently reviewed, adjudicated metadata that passes the agreement and licensing gates.

Never overwrite an annotator's original rows. Copy approved, non-sensitive metadata into a versioned release manifest only after the agreement gate passes.
