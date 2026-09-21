# UCA / UCF-Crime setup

The UCA examples use three videos from the UCF-Crime dataset:

```
data/uca/videos/Abuse001_x264.mp4
data/uca/videos/Abuse002_x264.mp4
data/uca/videos/Abuse003_x264.mp4
```

The videos are not redistributed in this repository. Request/download
UCF-Crime from its official dataset source, accept its terms, and copy or
symlink those three files into `data/uca/videos/` without renaming them.

```bash
ln -s "/absolute/path/to/Abuse001_x264.mp4" data/uca/videos/Abuse001_x264.mp4
ln -s "/absolute/path/to/Abuse002_x264.mp4" data/uca/videos/Abuse002_x264.mp4
ln -s "/absolute/path/to/Abuse003_x264.mp4" data/uca/videos/Abuse003_x264.mp4
```

The committed `uca_gallery.jsonl`, `uca_captions.jsonl`, and
`uca_grounding.jsonl` manifests all reference these paths.
`annotation_excerpt.json` is the retained ground-truth slice for the same three
videos. Local MP4s are ignored by Git.
