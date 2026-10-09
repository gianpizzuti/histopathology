# Results for the LNBI journal version

One folder per experiment, written by `scripts/run_unit.py` / `scripts/launch.py`
(`raw/*.json`) and `scripts/aggregate.py` (summaries, LaTeX tables, PDF figures).
See the main README for the file formats.

| Folder | Experiment | Reviewer item |
|---|---|---|
| `e1_vit_matched/` | ViT-B/16 with the ResNet-18 protocol (3 splits × 3 seeds), ResNet-18 re-run as paired reference | R1.2, R2.4 |
| `e2_resnet50/` | ResNet-50 backbone, SimCLR + BYOL, same protocol as ResNet-18 | R2.2 |
| `e3_dinov2/` | DINOv2 ViT-S/14 and ViT-B/14, pretrained and frozen (no SSL on our data) | R2.3 |
| `e3_barlow/` | ResNet-18 with Barlow Twins, same protocol as SimCLR/BYOL | R2.3 |
| `e4_federated/` | Federated SimCLR ResNet-18 on PCam (FedAvg, 5/10 clients, IID and Dirichlet label skew α 0.5/0.1), paired with the centralised SimCLR of E1; `legacy08_control/`: conference federated evaluation vs ours | federated |
| `e6_ood/` | OOD confidence PCam ↔ PANDA of the E1-E3 encoders vs supervised ResNet-18 (scratch, ImageNet), before/after temperature scaling | OOD mechanism |
| `probe_robustness/` | E1-E4 and E6 with a converged, cross-validated logistic-regression probe vs the protocol probe | probe sensitivity |
| `splits/` | universe, splits and labelled subsets as image ids | all |
