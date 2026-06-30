# I24V selected highway clips

Eight 1080p MP4 clips from the **I24V** interstate video dataset (Gloudemans et al., WACV 2024 — *So you think you can track?*). They are a small **selected** subset of the full 234-camera release, named `highway1.mp4` … `highway8.mp4`, laid out as a linear corridor for future cross-camera / MTMC-style MMDS examples.

## Layout

```
data/i24v_traffic/
  README.md
  videos/
    highway1.mp4 … highway8.mp4
data/i24v_traffic_feed.jsonl   # one row per camera (Input target for examples)
```

## Feed manifest

[`data/i24v_traffic_feed.jsonl`](../i24v_traffic_feed.jsonl) mirrors [`data/campus_traffic_feed.jsonl`](../campus_traffic_feed.jsonl):

- `camera_id` — `cam-i24v-highway1` … `cam-i24v-highway8`
- `video.path` — `data/i24v_traffic/videos/highwayN.mp4`
- `adjacent_cameras` — chain graph (each camera links to its neighbors along the corridor)
- `recorded_at`, `fps`, `duration_sec`, `num_frames`, `location`, `width`, `height`

Clips 1–3 are ~80 s (~2411 frames); clips 4–8 are ~60 s (~1812 frames) at ~29.97 fps.

## Obtaining the full I24V dataset

The complete dataset (~1 TB, `.mkv` per camera) requires a free account at [i24motion.org/data](https://i24motion.org/data). Utilities live in [I24-MOTION/i24-video-dataset-utils](https://github.com/I24-MOTION/i24-video-dataset-utils).

If you need to re-import from a local download folder:

```bash
cp "/path/to/I24V dataset (WACV 2024)-selected"/highway*.mp4 data/i24v_traffic/videos/
```

## Citation

```bibtex
@inproceedings{gloudemans2024so,
  title={So you think you can track?},
  author={Gloudemans, Derek and Zach{\'a}r, Gergely and Wang, Yanbing and Ji, Junyi and Nice, Matt and Bunting, Matt and Barbour, William W and Sprinkle, Jonathan and Piccoli, Benedetto and Monache, Maria Laura Delle and others},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={4528--4538},
  year={2024}
}
```

## Git

Video files under `data/i24v_traffic/videos/` are gitignored (~350 MB total). Clone this repo and copy the clips locally, or symlink from your I24V download directory.
