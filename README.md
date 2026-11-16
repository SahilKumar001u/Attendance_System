# AI-Powered Attendance System

Computer Vision project, Assignment 3. The webcam scans the classroom, recognises
registered students and writes the attendance log to a CSV file automatically.

- **Face detection:** YuNet (`cv2.FaceDetectorYN`)
- **Face recognition:** SFace (`cv2.FaceRecognizerSF`), 128-d embeddings compared by cosine similarity
- **Database:** `students/<name>/*.jpg`
- **Attendance log:** `attendance/YYYY-MM-DD.csv` (`name,date,time`)

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The two model files (about 37 MB) are downloaded into `models/` on first run.

## Usage

Register each student (at least 10). Look at the camera, press SPACE five times
with slightly different head angles, `q` to stop early:

```bash
.venv/bin/python attendance.py register "Sahil Kumar"
```

You can also just drop photos (one face per photo) into `students/<name>/`.

Take attendance. Green box = recognised, red box = unknown. Press `q` to finish;
the present/absent list is printed and the CSV is saved:

```bash
.venv/bin/python attendance.py scan
```

Use a classroom photo instead of the webcam (annotated result saved as `attendance/last_scan.jpg`):

```bash
.venv/bin/python attendance.py scan --image classroom.jpg
```

Accuracy report on the registered photos:

```bash
.venv/bin/python attendance.py evaluate
```

If the wrong camera opens, add `--camera 1` before the command name.

## How it works

1. Every registered photo is turned into a face embedding at start-up.
2. For each webcam frame, every detected face is aligned, embedded and compared
   with all registered embeddings.
3. If the best cosine similarity is at least 0.363 (the threshold OpenCV
   recommends for SFace), the face gets that student's name; otherwise "Unknown".
4. A student is marked present after matching in 3 frames, which filters out
   one-frame mistakes. Each student is written to the day's CSV only once.

## Evaluation

`evaluate` runs two tests on the registered photos:

- **Leave-one-out:** each photo is matched against all the *other* photos. It
  reports correct recognitions, false positives (recognised as a different
  student) and false negatives (not recognised at all).
- **Impostor test:** each student's own photos are hidden, so they act as an
  unregistered person. It reports how often they are still accepted as someone.

Results (fill in after registering the class and running `evaluate`):

| Metric | Result |
|---|---|
| Students / photos | |
| Correctly recognised | |
| False positives | |
| False negatives | |
| Unregistered person accepted | |

## Limitations

- Small or distant faces (back rows) are detected less reliably; a laptop webcam
  covers only the first few rows.
- Strong side poses, masks, hands over the face and poor or back lighting lower the similarity score.
- No liveness check: a printed photo or a phone screen showing a student's face is accepted.
- Identical twins and very similar faces can be confused.
- Accuracy depends on the registration photos; five varied photos per student work much better than one.
- Evaluation uses the registration photos themselves, taken in one sitting, so
  real-classroom accuracy will be somewhat lower than the reported figure.
