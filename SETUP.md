# FaceNet-PyTorch Setup Guide

## Step 1: Create Conda Environment

```bash
# Create new conda environment with Python 3.10
conda create -n face_attendance python=3.10 -y

# Activate the environment
conda activate face_attendance
```

## Step 2: Install PyTorch

**For CPU (recommended for most users):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**For GPU (if you have NVIDIA GPU with CUDA):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Step 3: Install Face Recognition Dependencies

```bash
pip install facenet-pytorch streamlit opencv-python numpy pandas pillow openpyxl sqlalchemy
```

## Step 4: Verify Installation

```bash
python -c "from facenet_pytorch import MTCNN, InceptionResnetV1; print('FaceNet-PyTorch installed successfully!')"
```

## Step 5: Run the Application

```bash
streamlit run app.py
```

## Complete One-Line Setup (Copy-Paste)

```bash
conda create -n face_attendance python=3.10 -y && conda activate face_attendance && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && pip install facenet-pytorch streamlit opencv-python numpy pandas pillow openpyxl sqlalchemy
```

## Troubleshooting

### Issue: "conda: command not found"
- Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)

### Issue: Camera not working
- Grant camera permissions in browser
- Close other apps using webcam
- Try Chrome browser

### Issue: Slow performance
- Use CPU version of PyTorch (already recommended above)
- Close other applications
- Reduce number of students if needed

## System Requirements

- Python 3.10
- ~2GB disk space
- Webcam
- Windows/Mac/Linux
- **No C++ Build Tools required!**
