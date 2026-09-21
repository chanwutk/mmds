# Data

This directory primarily commits small JSON/JSONL manifests, annotations, and
hand-labeled ground truth. Most dataset media is excluded because it is large
and may have separate access or licensing terms. The two clips required by the
cross-camera example (`highway2.mp4` and `highway3.mp4`) remain bundled.

## Local datasets

- [I24V traffic](i24v_traffic/README.md): `highway2.mp4` and `highway3.mp4`
  are committed for the cross-camera examples. Install the other six clips
  under `data/i24v_traffic/videos/` only when using the full corridor manifest.
- [UCA / UCF-Crime](uca/README.md): install the Abuse001/002/003 videos under
  `data/uca/videos/`.
- [Campus traffic](campus_traffic/README.md): install the synchronized seq3
  videos and, when using frame manifests, the extracted PNG frames.

The optional-media directories contain `.gitkeep` placeholders and their media
patterns are ignored. Copy or symlink authorized local files into place; do not
commit additional dataset media.

## Committed inputs

Small public-URL example inputs such as `animals.jsonl`, `clips.jsonl`, and the
NBA manifests run without local media. Dataset-specific feed, caption, and frame
manifests also remain committed so their expected schemas and paths are visible.
`i24v_traffic_highway2_highway3_5s_ground_truth.json` is hand-labeled ground
truth and remains versioned.

Generated outputs, including
`i24v_traffic_highway2_highway3_5s_nmsed.json`, are not source data and should
stay untracked.
