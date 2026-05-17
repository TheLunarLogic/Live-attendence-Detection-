import cv2
import numpy as np
import torch
import sys
from face_utils import load_face_models, crop_face, preprocess_face_for_embedding, normalize_embedding

def test_pipeline():
    print("=== Cross-Environment Embedding Pipeline Test ===")
    
    # 1. Generate a deterministic dummy "face" (e.g., gradient image)
    # This guarantees the exact same input bytes across all environments
    h, w = 200, 200
    dummy_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            dummy_bgr[i, j, 0] = (i + j) % 255       # B
            dummy_bgr[i, j, 1] = (i * 2) % 255       # G
            dummy_bgr[i, j, 2] = (j * 3) % 255       # R
            
    # Mock bounding box in the center
    box = [50, 50, 150, 150]
    
    print("1. Cropping dummy face with margin=0.15...")
    crop = crop_face(dummy_bgr, box, margin=0.15)
    
    if crop is None:
        print("FAIL: crop_face returned None")
        return
        
    print(f"   Crop shape: {crop.shape}, dtype: {crop.dtype}")
    print(f"   Crop mean: {crop.mean():.4f}, std: {crop.std():.4f}")
    
    print("2. Preprocessing face for embedding...")
    tensor = preprocess_face_for_embedding(crop)
    
    if tensor is None:
        print("FAIL: preprocess_face_for_embedding returned None")
        return
        
    print(f"   Tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
    print(f"   Tensor mean: {tensor.mean().item():.6f}, std: {tensor.std().item():.6f}")
    
    print("3. Loading models...")
    mtcnn, resnet, device = load_face_models()
    if resnet is None:
        print("FAIL: Could not load models")
        return
        
    print(f"   Device: {device}")
    
    print("4. Extracting embedding...")
    with torch.no_grad():
        tensor = tensor.to(device)
        emb = resnet(tensor).cpu().numpy()[0]
        
    print(f"   Raw embedding min: {emb.min():.6f}, max: {emb.max():.6f}")
    
    print("5. Normalizing embedding...")
    norm_emb = normalize_embedding(emb)
    
    # Print a checksum-like summary of the embedding
    emb_sum = norm_emb.sum()
    emb_mean = norm_emb.mean()
    emb_std = norm_emb.std()
    
    print("\n=== FINAL EMBEDDING CHECKSUM ===")
    print(f"Sum : {emb_sum:.8f}")
    print(f"Mean: {emb_mean:.8f}")
    print(f"Std : {emb_std:.8f}")
    print("================================")
    print("Run this script on both localhost and server.")
    print("The FINAL EMBEDDING CHECKSUM values must be IDENTICAL.")
    print("If they differ, there is a PyTorch/CUDA precision mismatch or library version mismatch.")

if __name__ == "__main__":
    test_pipeline()
