# Driver Drowsiness Detection

A real-time computer vision system that monitors a driver's face through a webcam and detects signs of drowsiness using three complementary methods: eye closure (EAR), yawning (MAR), and head-tilt estimation. Triggers an audio alarm when fatigue is detected.

## Tech Stack

| | |
|---|---|
| Language | Python |
| Vision | OpenCV, dlib (68-point facial landmarks) |
| Math | NumPy, SciPy, imutils |
| Audio | playsound |

## Features

- **Eye Aspect Ratio (EAR)** — Detects eye closure; triggers DROWSINESS alert if eyes are closed for consecutive frames
- **Mouth Aspect Ratio (MAR)** — Detects yawning when MAR exceeds threshold
- **Head Pose Estimation** — Measures forward head tilt using Perspective-n-Point (PnP) solve
- **Real-Time Display** — Bounding boxes, facial landmarks, status panel with EAR/MAR/Head Tilt values
- **Audio Alarm** — Plays `alarm.mp3` via background thread when drowsiness is detected
- **Auto Camera Detection** — Tries camera indices 0, 1, 2 automatically

## Quick Start

```bash
# Install dependencies
pip install -r Requirements.txt

# Run the detection
python ddd.py
```

Press **`q`** to quit the application.

## How It Works

| Component | File | Method |
|---|---|---|
| Eye Closure | `EAR.py` | Ratio of vertical to horizontal eye landmark distances (< 0.25 = closed) |
| Yawning | `MAR.py` | Ratio of vertical to horizontal mouth landmark distances (> 0.79 = yawning) |
| Head Tilt | `HeadPose.py` | 3D face model + solvePnP to estimate forward tilt angle |
| Main Loop | `ddd.py` | Webcam capture, face detection, metric computation, alarm trigger |

## Parameters

| Parameter | Threshold | Consecutive Frames |
|---|---|---|
| EAR (Eye Closure) | < 0.25 | 3+ frames |
| MAR (Yawning) | > 0.79 | Instant |
| Head Tilt | Forward tilt angle | Instant |

## Model

The system uses dlib's pre-trained **68-point facial landmark predictor** (`shape_predictor_68_face_landmarks.dat`) based on the iBUG 300-W dataset.
