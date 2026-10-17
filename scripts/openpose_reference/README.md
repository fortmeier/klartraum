# OpenPose reference drawing

Generates the fixture for `tests/test_openpose_skeleton.cpp`: two COCO-18
skeletons drawn with the body-pose convention of ControlNet's OpenPose
annotator (`draw_bodypose` in lllyasviel/ControlNet and controlnet_aux), which
is what ControlNet-OpenPose models were trained on. Klartraum's
`OpenPoseSkeletonRenderer` reproduces this drawing on the GPU.

From this directory:

```bash
uv run python generate_reference.py
```

It writes `data/openpose/reference_keypoints.txt` (normalized keypoints and the
stick width and joint radius used) and `data/openpose/reference_256.ppm`.
