# Face Recognition Attendance System

A production-ready face recognition attendance system for college teachers.  
Built with **FaceNet-PyTorch** (MTCNN + InceptionResnetV1) — no C++ build tools required on Windows.

---

## Features

| Feature | Description |
|---------|-------------|
| 👨‍🎓 Manual Registration | Register students via webcam (5 samples each) |
| 📸 Real-Time Attendance | Automatic YOLO-based webcam attendance |
| 📂 Dataset Registration ✨ | Bulk-register a whole class from student images with filename parsing |
| 🎬 Video Attendance ✨ | Upload a classroom video — system marks attendance automatically |
| 📥 Excel Export | Export any attendance report to `.xlsx` |

---

## Technology Stack

- **Frontend**: Streamlit  
- **Face Detection**: MTCNN  
- **Face Recognition**: InceptionResnetV1 (VGGFace2 pretrained)  
- **Deep Learning**: PyTorch  
- **Object Detection**: YOLOv8 (real-time mode)  

- **Database**: SQLite (JSON-serialized embeddings)  
- **Export**: Pandas + openpyxl  

---

## Project Structure

```
face_attendance/
│
├── app.py                   # Main Streamlit app (5 pages)
├── database.py              # SQLite CRUD operations
├── face_utils.py            # MTCNN + FaceNet utilities
├── attendance.py            # Cosine-similarity matching + Excel export
├── auto_attendance_v2.py    # YOLO real-time attendance
├── dataset_registration.py  # [NEW] Bulk filename-based registration
├── video_attendance.py      # [NEW] Video-based attendance pipeline
├── requirements.txt
├── SETUP.md                 # Conda environment guide
└── data/
    ├── attendance.db        # SQLite database (auto-created)
    └── exports/             # Excel exports (auto-created)
```

---

## Quick Start

### 1 — Create Conda Environment (Recommended)

```bash
conda create -n torch_gpu python=3.10 -y
conda activate torch_gpu
```

### 2 — Install PyTorch (CPU)

```bash
conda install -c pytorch pytorch torchvision torchaudio cpuonly -y
```

### 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### 5 — Run

```bash
streamlit run app.py
```

---

## Usage Guide

### Manual Student Registration
1. Create or select a class
2. Enter roll number + name
3. Capture 5 webcam photos
4. Click **Register Student**

### Dataset Registration (Bulk) 📂
1. Prepare a ZIP of student images  
   Each image must contain **only the student's face** (no text).
   Encode the roll number + name in the filename (e.g., `Name_RollNo.jpg`).
2. Select class → **Register from Dataset**
3. Upload ZIP
4. Click **Start Registration**

Expected image filename format:
```
Rahul_Sharma_221045.jpg
```
The system generates 4 augmented copies per image to improve recognition accuracy.

### Real-Time Attendance 📸
1. Select class → **Take Attendance**
2. Click **Start Automatic Attendance**
3. Face the webcam toward the class

### Video Attendance 🎬
1. Select class → **Attendance from Video**
2. Upload `.mp4` / `.avi` / `.mov` file
3. Adjust frame-skip and similarity threshold sliders
4. Click **Process Video**
5. Download the Excel export

---

## Database Schema

```sql
-- Classes
CREATE TABLE classes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    class_name  TEXT UNIQUE NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Students
CREATE TABLE students (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    class_id    INTEGER NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
    roll_no     TEXT NOT NULL,
    name        TEXT NOT NULL,
    embedding   TEXT NOT NULL,   -- JSON array, 512-d float
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(class_id, roll_no)
);
```

---

## Configuration

| Parameter | Default | Where to change |
|-----------|---------|----------------|
| Similarity threshold | 0.6 | `attendance.py` or UI slider |
| Min samples (manual) | 5 | `app.py` |
| Augmented copies (dataset) | 4 | `dataset_registration.py` |
| Frame skip (video) | 10 | UI slider |


---

## System Requirements

- Python 3.8 – 3.11  
- Webcam (for manual / real-time modes)  
- ~1.5 GB disk space  
- **No C++ Build Tools required** ✅  
- Internet connection on first run (model download ~100 MB)  

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Camera not working | Grant browser permissions; close other apps using webcam |
| Filenames not parsing names/roll numbers | Format filenames as Name_RollNo.jpg |
| Poor recognition accuracy | Lower threshold slider; ensure good lighting during registration |
| Video processing slow | Increase frame-skip slider; close other applications |
| `facenet-pytorch` install fails | Use Conda environment (see SETUP.md) |
