"""AI-powered attendance: register student faces, scan the class, log to CSV."""
import argparse
import csv
import os
import sys
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # OpenCV 5 prints a harmless warning per model
import cv2
import numpy as np

ROOT = Path(__file__).parent
STUDENTS, LOGS, MODELS = ROOT / "students", ROOT / "attendance", ROOT / "models"
ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models/"
DET = "face_detection_yunet/face_detection_yunet_2023mar.onnx"
REC = "face_recognition_sface/face_recognition_sface_2021dec.onnx"
THRESHOLD = 0.363  # cosine similarity cut-off OpenCV recommends for SFace
CONFIRM_FRAMES = 3  # a face must match in this many frames before it is marked
GREEN, RED = (0, 200, 0), (0, 0, 255)


def model(rel):
    path = MODELS / Path(rel).name
    if not path.exists():
        MODELS.mkdir(exist_ok=True)
        print("Downloading", path.name)
        urllib.request.urlretrieve(ZOO + rel, path)
    return str(path)


detector = cv2.FaceDetectorYN.create(model(DET), "", (320, 320), score_threshold=0.8)
recognizer = cv2.FaceRecognizerSF.create(model(REC), "")


def faces(img):
    """Return [(box, unit embedding)] for every face in a BGR image."""
    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    _, found = detector.detect(img)
    out = []
    for f in found if found is not None else []:
        emb = recognizer.feature(recognizer.alignCrop(img, f)).flatten()
        out.append((f[:4].astype(int), emb / np.linalg.norm(emb)))
    return out


def load_db():
    """Read students/<name>/*.jpg into (names, embedding matrix)."""
    names, embs = [], []
    for p in sorted(STUDENTS.glob("*/*")):
        img = cv2.imread(str(p))
        found = faces(img) if img is not None else []
        if len(found) != 1:
            print(f"Skipping {p}: expected 1 face, found {len(found)}")
            continue
        names.append(p.parent.name)
        embs.append(found[0][1])
    if not names:
        sys.exit('No registered students. Run: python attendance.py register "Name"')
    return names, np.array(embs)


def identify(emb, names, embs):
    """Return (name or None if unknown, similarity)."""
    sims = embs @ emb
    i = int(sims.argmax())
    return (names[i] if sims[i] >= THRESHOLD else None), float(sims[i])


def mark(name, log):
    """Append name to the log once; return True if newly marked."""
    log.parent.mkdir(exist_ok=True)
    with log.open("a+", newline="") as f:
        f.seek(0)
        rows = list(csv.reader(f))
        if any(r and r[0] == name for r in rows):
            return False
        w = csv.writer(f)
        if not rows:
            w.writerow(["name", "date", "time"])
        now = datetime.now()
        w.writerow([name, f"{now:%Y-%m-%d}", f"{now:%H:%M:%S}"])
    return True


def open_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {index}. Try --camera 1")
    return cap


def label(img, box, text, color):
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, text, (x, max(y - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def register(args):
    if Path(args.name).name != args.name or args.name.startswith("."):
        sys.exit("Name must be a plain name, e.g. \"Sahil Kumar\"")
    folder = STUDENTS / args.name
    folder.mkdir(parents=True, exist_ok=True)
    n = len(list(folder.glob("*.jpg")))
    target = n + args.photos
    cap = open_camera(args.camera)
    while n < target:
        ok, frame = cap.read()
        if not ok:
            break
        found = faces(frame)
        view = frame.copy()
        for box, _ in found:
            label(view, box, args.name, GREEN if len(found) == 1 else RED)
        cv2.putText(view, f"SPACE = capture ({n}/{target})   q = quit", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
        cv2.imshow("Register", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and len(found) == 1:  # exactly one face, so we know whose it is
            cv2.imwrite(str(folder / f"{n:02d}.jpg"), frame)
            n += 1
    cap.release()
    cv2.destroyAllWindows()
    print(f"{args.name}: {n} photo(s) in {folder}")


def scan(args):
    names, embs = load_db()
    log = LOGS / f"{datetime.now():%Y-%m-%d}.csv"
    seen = Counter()

    def process(frame, need):
        for box, emb in faces(frame):
            name, sim = identify(emb, names, embs)
            label(frame, box, f"{name or 'Unknown'} {sim:.2f}", GREEN if name else RED)
            if name:
                seen[name] += 1
                if seen[name] >= need and mark(name, log):
                    print("Present:", name)

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit(f"Cannot read image {args.image}")
        process(frame, 1)
        LOGS.mkdir(exist_ok=True)
        cv2.imwrite(str(LOGS / "last_scan.jpg"), frame)
    else:
        cap = open_camera(args.camera)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            process(frame, CONFIRM_FRAMES)
            cv2.putText(frame, "q = finish", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
            cv2.imshow("Attendance", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

    if log.exists():
        with log.open(newline="") as f:
            present = {r[0] for r in list(csv.reader(f))[1:] if r}
    else:
        present = set()
    absent = sorted(set(names) - present)
    print(f"\nPresent ({len(present)}): {', '.join(sorted(present)) or '-'}")
    print(f"Absent  ({len(absent)}): {', '.join(absent) or '-'}")
    print("Log:", log)


def evaluate(_):
    """Leave-one-out accuracy on the registered photos, plus an impostor test."""
    names, embs = load_db()
    names = np.array(names)
    same = names[:, None] == names[None, :]
    sims = embs @ embs.T

    loo = sims.copy()
    np.fill_diagonal(loo, -1)  # a photo may not match itself
    best, accepted = loo.argmax(1), loo.max(1) >= THRESHOLD
    correct = accepted & (names[best] == names)
    wrong = accepted & (names[best] != names)

    # impostor test: hide each student's own photos, so they are "unregistered"
    impostor = np.where(same, -1, sims).max(1) >= THRESHOLD

    n = len(names)
    print(f"Students: {len(set(names))}   photos: {n}   threshold: {THRESHOLD}")
    print(f"Correctly recognised:                 {correct.sum()}/{n} ({correct.mean():.1%})")
    print(f"False positives (marked as another):  {wrong.sum()}/{n} ({wrong.mean():.1%})")
    print(f"False negatives (not recognised):     {(~accepted).sum()}/{n} ({(~accepted).mean():.1%})")
    print(f"Unregistered person accepted:         {impostor.sum()}/{n} ({impostor.mean():.1%})")
    for i in np.flatnonzero(wrong):
        print(f"  confused {names[i]} with {names[best[i]]} ({loo[i, best[i]]:.2f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register", help="capture photos of one student from the webcam")
    r.add_argument("name")
    r.add_argument("--photos", type=int, default=5)
    r.set_defaults(func=register)
    s = sub.add_parser("scan", help="recognise faces and mark attendance")
    s.add_argument("--image", help="use a classroom photo instead of the webcam")
    s.set_defaults(func=scan)
    sub.add_parser("evaluate", help="accuracy report on registered photos").set_defaults(func=evaluate)
    args = ap.parse_args()
    args.func(args)
