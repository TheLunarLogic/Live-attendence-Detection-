"""
Face recognition utilities using FaceNet-PyTorch.
Handles face detection (MTCNN) and embedding extraction (InceptionResnetV1).
"""

import cv2
import numpy as np
import streamlit as st
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
from typing import List, Optional, Tuple, Union
from PIL import Image

# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
EMBEDDING_PIPELINE_VERSION = "v2"
_FACENET_INPUT_SIZE = 160
_CROP_MARGIN = 0.15

# Ensure exact deterministic math on CUDA
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

@st.cache_resource
def load_face_models():
    """
    Load and cache FaceNet-PyTorch models.
    Uses MTCNN for face detection and InceptionResnetV1 for embeddings.
    
    Returns:
        Tuple of (mtcnn, resnet) models
    """
    try:
        # Check if CUDA is available
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Initialize MTCNN for face detection (margin not critical here since we only use it for boxes now)
        mtcnn = MTCNN(
            image_size=_FACENET_INPUT_SIZE,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=device,
            keep_all=True  # Detect multiple faces
        )
        
        # Initialize InceptionResnetV1 for face recognition
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        
        return mtcnn, resnet, device
    
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

def crop_face(image_bgr: np.ndarray, box: Union[list, np.ndarray], margin: float = _CROP_MARGIN) -> Optional[np.ndarray]:
    """
    Deterministically crop a face from a BGR image using a consistent margin.
    """
    try:
        x1, y1, x2, y2 = [int(c) for c in box]
        h, w = image_bgr.shape[:2]
        
        # Calculate margin in pixels
        margin_px = int(max(x2 - x1, y2 - y1) * margin)
        
        x1m, y1m = max(0, x1 - margin_px), max(0, y1 - margin_px)
        x2m, y2m = min(w, x2 + margin_px), min(h, y2 + margin_px)
        
        crop = image_bgr[y1m:y2m, x1m:x2m]
        
        if crop.size == 0:
            return None
        return crop
    except Exception:
        return None

def preprocess_face_for_embedding(face_bgr: np.ndarray) -> Optional[torch.Tensor]:
    """
    Centralized, deterministic preprocessing pipeline.
    Ensures embeddings are strictly identical across environments.
    """
    try:
        # 1. BGR -> RGB
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        
        # 2. Resize EXACTLY 160x160 using INTER_AREA for robust downscaling
        face_resized = cv2.resize(face_rgb, (_FACENET_INPUT_SIZE, _FACENET_INPUT_SIZE), interpolation=cv2.INTER_AREA)
        
        # 3. Convert np.float32
        face_float = face_resized.astype(np.float32)
        
        # 4. Normalize using facenet-pytorch's exact standard
        # fixed_image_standardization literally does (x - 127.5)/128.0 but this is safest
        face_normalized = fixed_image_standardization(face_float)
        
        # 5. Convert tensor: shape = (1, 3, 160, 160)
        face_tensor = torch.from_numpy(face_normalized).permute(2, 0, 1).unsqueeze(0)
        
        # 6. Ensure contiguous memory for deterministic CUDA access
        face_tensor = face_tensor.contiguous()
        
        return face_tensor
    except Exception:
        return None


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    L2 normalize embedding vector.
    
    Args:
        embedding: Raw embedding vector
    
    Returns:
        Normalized embedding
    """
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def extract_embedding(image: np.ndarray, mtcnn, resnet, device) -> Tuple[bool, Optional[np.ndarray], str]:
    """
    Extract face embedding from a single image.
    
    Args:
        image: Input image (numpy array in RGB format)
        mtcnn: MTCNN model for face detection
        resnet: InceptionResnetV1 model for embedding extraction
        device: torch device (CPU or CUDA)
    
    Returns:
        Tuple of (success, normalized_embedding, message)
    """
    if mtcnn is None or resnet is None:
        return False, None, "Face models not loaded"
    
    try:
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Detect faces (detect returns bounding boxes)
        boxes, probs = mtcnn.detect(pil_image)
        
        if boxes is None or len(boxes) == 0:
            return False, None, "No face detected in the image"
        
        # Filter by minimum confidence if needed, but here we just take the first
        valid_indices = [i for i, p in enumerate(probs) if p is not None and p >= 0.90]
        if not valid_indices:
            # Fallback to the highest probability face if none meet the strict threshold
            valid_indices = [0]
            
        if len(valid_indices) > 1:
            return False, None, f"Multiple faces detected ({len(valid_indices)}). Please ensure only one face is visible"
        
        # Get the detected face box
        best_idx = valid_indices[0]
        box = boxes[best_idx]
        
        # Convert RGB image to BGR for our centralized pipeline
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Crop the face using standardized margin
        face_crop_bgr = crop_face(image_bgr, box)
        if face_crop_bgr is None:
            return False, None, "Failed to crop face"
            
        # Preprocess using centralized pipeline
        face_tensor = preprocess_face_for_embedding(face_crop_bgr)
        if face_tensor is None:
            return False, None, "Failed to preprocess face"
            
        # Extract embedding
        with torch.no_grad():
            face_tensor = face_tensor.to(device)
            embedding = resnet(face_tensor).cpu().numpy()[0]
        
        # Normalize embedding
        normalized_embedding = normalize_embedding(embedding)
        
        return True, normalized_embedding, "Face embedding extracted successfully"
    
    except Exception as e:
        return False, None, f"Error extracting embedding: {str(e)}"


def extract_multi_sample_embedding(images: List[np.ndarray], mtcnn, resnet, device) -> Tuple[bool, Optional[np.ndarray], str]:
    """
    Extract and average embeddings from multiple samples for robustness.
    
    Args:
        images: List of input images (numpy arrays in RGB format)
        mtcnn: MTCNN model
        resnet: InceptionResnetV1 model
        device: torch device
    
    Returns:
        Tuple of (success, averaged_normalized_embedding, message)
    """
    if not images or len(images) == 0:
        return False, None, "No images provided"
    
    embeddings = []
    failed_count = 0
    
    for idx, image in enumerate(images):
        success, embedding, msg = extract_embedding(image, mtcnn, resnet, device)
        
        if success:
            embeddings.append(embedding)
        else:
            failed_count += 1
    
    if len(embeddings) == 0:
        return False, None, "Failed to extract embeddings from all samples"
    
    # Average the embeddings
    avg_embedding = np.mean(embeddings, axis=0)
    
    # Normalize the averaged embedding
    normalized_avg_embedding = normalize_embedding(avg_embedding)
    
    success_count = len(embeddings)
    message = f"Successfully processed {success_count}/{len(images)} samples"
    
    if failed_count > 0:
        message += f" ({failed_count} failed)"
    
    return True, normalized_avg_embedding, message


def detect_faces(image: np.ndarray, mtcnn, resnet, device) -> Tuple[bool, List[np.ndarray], List[dict], str]:
    """
    Detect multiple faces in an image and extract their embeddings.
    Used for attendance taking from classroom images.
    
    Args:
        image: Input image (numpy array in RGB format)
        mtcnn: MTCNN model
        resnet: InceptionResnetV1 model
        device: torch device
    
    Returns:
        Tuple of (success, list_of_normalized_embeddings, list_of_face_info, message)
        face_info contains: bbox, probability
    """
    if mtcnn is None or resnet is None:
        return False, [], [], "Face models not loaded"
    
    try:
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Detect faces and get bounding boxes
        # Note: When keep_all=True, mtcnn returns (faces, probs) not (faces, probs, boxes)
        # We need to use detect method separately to get boxes
        boxes, probs = mtcnn.detect(pil_image)
        
        if boxes is None or len(boxes) == 0:
            return False, [], [], "No faces detected in the image"
        
        # Now extract the face tensors using centralized pipeline
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        embeddings = []
        face_infos = []
        
        # Extract embeddings for each detected face
        with torch.no_grad():
            for idx, box in enumerate(boxes):
                prob = probs[idx] if probs is not None else None
                
                # Store face information first
                face_info = {
                    'bbox': box.tolist() if box is not None else None,
                    'probability': float(prob) if prob is not None else None
                }
                
                # Crop and preprocess
                face_crop_bgr = crop_face(image_bgr, box)
                if face_crop_bgr is None:
                    continue
                    
                face_tensor = preprocess_face_for_embedding(face_crop_bgr)
                if face_tensor is None:
                    continue
                    
                # Extract embedding
                face_tensor = face_tensor.to(device)
                embedding = resnet(face_tensor).cpu().numpy()[0]
                
                # Normalize embedding
                normalized_embedding = normalize_embedding(embedding)
                
                embeddings.append(normalized_embedding)
                face_infos.append(face_info)
        
        if not embeddings:
            return False, [], [], "Faces detected but failed to process them"
            
        message = f"Detected {len(embeddings)} face(s) in the image"
        return True, embeddings, face_infos, message
    
    except Exception as e:
        return False, [], [], f"Error detecting faces: {str(e)}"


def pil_to_numpy(pil_image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to numpy array in RGB format.
    
    Args:
        pil_image: PIL Image object
    
    Returns:
        Numpy array in RGB format
    """
    return np.array(pil_image.convert('RGB'))


def validate_image(image: np.ndarray) -> Tuple[bool, str]:
    """
    Validate image quality and format.
    
    Args:
        image: Input image as numpy array
    
    Returns:
        Tuple of (is_valid, message)
    """
    if image is None:
        return False, "Image is None"
    
    if not isinstance(image, np.ndarray):
        return False, "Image must be a numpy array"
    
    if len(image.shape) != 3:
        return False, "Image must be a color image (3 channels)"
    
    if image.shape[2] != 3:
        return False, "Image must have 3 color channels (RGB)"
    
    # Check minimum size
    height, width = image.shape[:2]
    if height < 100 or width < 100:
        return False, "Image is too small (minimum 100x100 pixels)"
    
    return True, "Image is valid"
