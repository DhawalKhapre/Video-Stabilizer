# Video-Stabilizer

Video stabilization is an integral video improvement technique for reducing trembling or jittery motion from videos. The algorithm uses Feature extraction to extract optimal points of interest; Optical flow to track the points throughout the entire video; Trajectory tracking to plot the relative motion of the tracked points in the video; and Moving averages to smoothen the trajectory of the points when jitters are present; to stabilize the video.

### Prerequisites

```
OpenCV - 4.5.1.48
Matplotlib - 3.1.3
Pandas - 1.0.1
Numpy - 1.19.5
```

### How-to

1. Clone the repository.
2. Run `Video-Stabilizer-Python.py` or `Video-Stabilizer-Jupyter.ipynb`.
3. Enter video filename(with full path) when prompted.
