# Installation Guide for Windows

## Prerequisites

This project requires InsightFace, which needs special installation steps on Windows.

## Step-by-Step Installation

### Option 1: Using Pre-built Wheel (Recommended for Windows)

1. **Install basic dependencies:**
```bash
pip install streamlit opencv-python numpy pandas openpyxl onnxruntime pillow scikit-learn
```

2. **Install InsightFace without building:**
```bash
pip install insightface --no-build-isolation
```

If this fails, try:
```bash
pip install insightface --no-deps
pip install onnxruntime numpy opencv-python
```

### Option 2: Install Visual C++ Build Tools

If you prefer to build from source:

1. Download and install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. During installation, select "Desktop development with C++"
3. Then run:
```bash
pip install -r requirements.txt
pip install insightface
```

### Option 3: Use Conda (Easiest)

```bash
conda create -n attendance python=3.10
conda activate attendance
conda install -c conda-forge insightface
pip install streamlit pandas openpyxl
```

## Verify Installation

Run this to verify InsightFace is installed correctly:

```python
python -c "from insightface.app import FaceAnalysis; print('InsightFace installed successfully!')"
```

## Running the Application

Once all dependencies are installed:

```bash
streamlit run app.py
```

## Troubleshooting

### Error: "Microsoft Visual C++ 14.0 or greater is required"
- Use Option 1 or Option 3 above
- Or install Visual C++ Build Tools (Option 2)

### Error: "No module named 'insightface'"
- Make sure you've installed InsightFace using one of the methods above
- Try: `pip install insightface --no-build-isolation`

### Model Download Issues
- The buffalo_l model (~600MB) downloads on first run
- Ensure stable internet connection
- Check firewall settings

### Camera Not Working
- Grant camera permissions in browser
- Close other apps using the webcam
- Try a different browser (Chrome recommended)

## System Requirements

- Python 3.8 - 3.11 (3.13 may have compatibility issues with some packages)
- Windows 10/11
- Webcam
- ~2GB free disk space
- Internet connection (first run only)
