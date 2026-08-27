# Campus traffic setup

The committed campus manifests describe two synchronized views of sequence 3:
an infrastructure camera and a drone. Media is not committed.

Expected video paths:

```
data/campus_traffic/seq3-infra.mp4
data/campus_traffic/seq3-drone.mp4
```

Frame-level manifests additionally expect 50 images per view:

```
data/campus_traffic/infra/frame_000001.png ... frame_000050.png
data/campus_traffic/drone/frame_000001.png ... frame_000050.png
```

Copy or symlink the authorized seq3 source assets into these exact locations.
If your source consists of the PNG sequences, the videos can be rebuilt with
FFmpeg:

```bash
ffmpeg -framerate 30 -i data/campus_traffic/infra/frame_%06d.png \
  -c:v libx264 -pix_fmt yuv420p data/campus_traffic/seq3-infra.mp4
ffmpeg -framerate 30 -i data/campus_traffic/drone/frame_%06d.png \
  -c:v libx264 -pix_fmt yuv420p data/campus_traffic/seq3-drone.mp4
```

The PNGs and MP4s are ignored by Git. The feed and frame manifests remain
versioned as lightweight metadata.
