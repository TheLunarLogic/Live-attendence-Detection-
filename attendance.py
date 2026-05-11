"""
Attendance processing module.
Handles face matching, attendance generation, and Excel export.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple
import os


def match_faces(detected_embeddings: List[np.ndarray], 
                student_data: List[Dict], 
                threshold: float = 0.6) -> Tuple[List[Dict], List[int]]:
    """
    Match detected face embeddings with stored student embeddings using cosine similarity.
    
    Args:
        detected_embeddings: List of normalized embeddings from detected faces
        student_data: List of student dictionaries with 'id', 'roll_no', 'name', 'embedding'
        threshold: Similarity threshold for matching (default: 0.6)
    
    Returns:
        Tuple of (matched_students, unmatched_indices)
        matched_students: List of dicts with student info and similarity score
        unmatched_indices: Indices of detected faces that couldn't be matched
    """
    matched_students = []
    matched_student_ids = set()  # Prevent duplicate marking
    unmatched_indices = []
    
    for face_idx, detected_embedding in enumerate(detected_embeddings):
        best_match = None
        best_similarity = -1
        
        # Compare with all students
        for student in student_data:
            student_embedding = student['embedding']
            
            # Skip if student already matched
            if student['id'] in matched_student_ids:
                continue
            
            # Calculate cosine similarity using dot product (embeddings are normalized)
            similarity = np.dot(detected_embedding, student_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = student
        
        # Check if best match exceeds threshold
        if best_match is not None and best_similarity >= threshold:
            matched_students.append({
                'id': best_match['id'],
                'roll_no': best_match['roll_no'],
                'name': best_match['name'],
                'similarity': float(best_similarity),
                'face_index': face_idx
            })
            matched_student_ids.add(best_match['id'])
        else:
            unmatched_indices.append(face_idx)
    
    return matched_students, unmatched_indices


def generate_attendance_report(matched_students: List[Dict], 
                               all_students: List[Dict], 
                               date: str = None) -> pd.DataFrame:
    """
    Generate attendance report DataFrame.
    
    Args:
        matched_students: List of matched student dictionaries
        all_students: List of all students in the class
        date: Date string (defaults to current date)
    
    Returns:
        Pandas DataFrame with attendance report
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    # Create set of matched student IDs for quick lookup
    matched_ids = {student['id'] for student in matched_students}
    
    # Build attendance records
    attendance_records = []
    
    for student in all_students:
        status = "Present" if student['id'] in matched_ids else "Absent"
        
        # Find similarity score if present
        similarity = None
        if status == "Present":
            for matched in matched_students:
                if matched['id'] == student['id']:
                    similarity = matched['similarity']
                    break
        
        record = {
            'Roll No': student['roll_no'],
            'Name': student['name'],
            'Status': status,
            'Date': date
        }
        
        # Add similarity score for present students
        if similarity is not None:
            record['Confidence'] = f"{similarity:.3f}"
        else:
            record['Confidence'] = "-"
        
        attendance_records.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(attendance_records)
    
    # Sort by roll number
    df = df.sort_values('Roll No').reset_index(drop=True)
    
    return df


def export_to_excel(df: pd.DataFrame, class_name: str, date: str = None) -> Tuple[bool, str]:
    """
    Export attendance DataFrame to Excel file.
    
    Args:
        df: Attendance DataFrame
        class_name: Name of the class
        date: Date string (defaults to current date)
    
    Returns:
        Tuple of (success, file_path_or_error_message)
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    try:
        # Create exports directory if it doesn't exist
        export_dir = "data/exports"
        os.makedirs(export_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{class_name}_{date}_{timestamp}.xlsx"
        filepath = os.path.join(export_dir, filename)
        
        # Export to Excel
        df.to_excel(filepath, index=False, engine='openpyxl')
        
        return True, filepath
    
    except Exception as e:
        return False, f"Error exporting to Excel: {str(e)}"


def calculate_attendance_statistics(df: pd.DataFrame) -> Dict:
    """
    Calculate attendance statistics from DataFrame.
    
    Args:
        df: Attendance DataFrame
    
    Returns:
        Dictionary with statistics
    """
    total_students = len(df)
    present_count = len(df[df['Status'] == 'Present'])
    absent_count = len(df[df['Status'] == 'Absent'])
    
    attendance_percentage = (present_count / total_students * 100) if total_students > 0 else 0
    
    return {
        'total_students': total_students,
        'present': present_count,
        'absent': absent_count,
        'attendance_percentage': attendance_percentage
    }


def get_unknown_faces_info(unmatched_indices: List[int], 
                          face_infos: List[Dict]) -> List[Dict]:
    """
    Get information about unmatched (unknown) faces.
    
    Args:
        unmatched_indices: List of indices of unmatched faces
        face_infos: List of face information dictionaries
    
    Returns:
        List of unknown face information
    """
    unknown_faces = []
    
    for idx in unmatched_indices:
        if idx < len(face_infos):
            unknown_faces.append({
                'index': idx,
                'bbox': face_infos[idx].get('bbox'),
                'probability': face_infos[idx].get('probability')
            })
    
    return unknown_faces
