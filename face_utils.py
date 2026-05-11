"""
Face recognition utilities using FaceNet-PyTorch.
Handles face detection (MTCNN) and embedding extraction (InceptionResnetV1).
"""

import cv2
import numpy as np
import streamlit as st
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from typing import List, Optional, Tuple
from PIL import Image


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
        
        # Initialize MTCNN for face detection
        mtcnn = MTCNN(
            image_size=160,
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
        st.error(f"Error loading face models: {str(e)}")
        return None, None, None


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
        
        # Detect faces
        faces, probs = mtcnn(pil_image, return_prob=True)
        
        if faces is None or len(faces) == 0:
            return False, None, "No face detected in the image"
        
        if len(faces) > 1:
            return False, None, f"Multiple faces detected ({len(faces)}). Please ensure only one face is visible"
        
        # Get the detected face
        face = faces[0]
        
        # Extract embedding
        with torch.no_grad():
            face = face.unsqueeze(0).to(device)  # Add batch dimension
            embedding = resnet(face).cpu().numpy()[0]
        
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
        
        # Now extract the face tensors
        faces = mtcnn(pil_image)
        
        if faces is None:
            return False, [], [], "No faces detected in the image"
        
        # Handle single face case (mtcnn returns tensor instead of list)
        if len(faces.shape) == 3:
            faces = faces.unsqueeze(0)
        
        embeddings = []
        face_infos = []
        
        # Extract embeddings for each detected face
        with torch.no_grad():
            for idx in range(len(faces)):
                # Extract embedding
                face_tensor = faces[idx].unsqueeze(0).to(device)
                embedding = resnet(face_tensor).cpu().numpy()[0]
                
                # Normalize embedding
                normalized_embedding = normalize_embedding(embedding)
                embeddings.append(normalized_embedding)
                
                # Store face information
                face_info = {
                    'bbox': boxes[idx].tolist() if boxes is not None else None,
                    'probability': float(probs[idx]) if probs is not None else None
                }
                face_infos.append(face_info)
        
        message = f"Detected {len(faces)} face(s) in the image"
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
