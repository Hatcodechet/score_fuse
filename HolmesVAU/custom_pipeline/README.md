# Custom HolmesVAU Pipeline

This folder contains an isolated pipeline that keeps the original HolmesVAU project untouched.

Default flow:

1. Match `.mp4` videos under `/workspace/test` with `__0.npy` UCF feature files.
2. Infer snippet-level anomaly scores with `/workspace/VHung/model/model_ucf.pth`.
3. Run ATS-style density-aware sampling on those scores.
4. Map sampled snippet indices to real video frames with stride `16`.
5. Generate a HolmesVAU description from the selected real frames.
6. Save scores, sampled indices, description, and visualizations under `outputs/<video_name>/`.

Primary entry point:

```bash
cd /workspace/score_fuse/HolmesVAU/custom_pipeline
python3 run_full_pipeline.py \
  --video-root /workspace/test \
  --feature-root /workspace/VHung/data/UCFClipFeatures \
  --checkpoint /workspace/VHung/model/model_ucf.pth \
  --holmes-model-path /workspace/HolmesVAU-2B
```
