# I24V selected highway clips

The cross-camera vehicle join uses the first five seconds of the adjacent
`highway2.mp4` and `highway3.mp4` clips from the I24V interstate video dataset.
The manifest is committed at
`data/i24v_traffic_highway2_highway3_5s.jsonl`; the media files are not.

After obtaining the dataset under its terms from
[i24motion.org/data](https://i24motion.org/data), copy or symlink the two clips
into:

```text
data/i24v_traffic/videos/highway2.mp4
data/i24v_traffic/videos/highway3.mp4
```

The tests use synthetic track rows and do not read these videos.

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
