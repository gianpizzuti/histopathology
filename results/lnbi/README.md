# Results for the LNBI journal version

One folder per experiment, written by `scripts/run_unit.py` / `scripts/launch.py`
(`raw/*.json`) and `scripts/aggregate.py` (summaries, LaTeX tables, PDF figures).
See the main README for the file formats.

| Folder | Experiment | Reviewer item |
|---|---|---|
| `e1_vit_matched/` | ViT-B/16 with the ResNet-18 protocol (3 splits × 3 seeds), ResNet-18 re-run as paired reference | R1.2, R2.4 |
| `e2_resnet50/` | ResNet-50 backbone, SimCLR + BYOL, same protocol as ResNet-18 | R2.2 |
| `splits/` | universe, splits and labelled subsets as image ids | all |
