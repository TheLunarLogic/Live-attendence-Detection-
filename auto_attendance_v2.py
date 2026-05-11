"""
Auto Attendance V2 - Streamlit Native Display
Displays video feed directly in Streamlit instead of using cv2.imshow
"""

import cv2
import numpy as np
import time
import streamlit as st
from typing import List, Dict
from database import Database
from face_utils import extract_embedding
from ultralytics import YOLO
import winsound
from datetime import datetime


class EmbeddingCache:
    """Cache for student embeddings"""
    def __init__(self, db: Database, class_id: int):
        self.embeddings = []
        self.load_from_db(db, class_id)
    
    def load_from_db(self, db: Database, class_id: int):
        """Load all student embeddings for the class"""
        students = db.get_students_by_class(class_id)
        for student in students:
            if student['embedding'] is not None:
                self.embeddings.append({
                    'id': student['id'],
                    'roll_no': student['roll_no'],
                    'name': student['name'],
                    'embedding': np.array(student['embedding'])
                })
    
    def find_match(self, query_embedding: np.ndarray, threshold: float = 0.6):
        """Find best matching student"""
        if len(self.embeddings) == 0:
            return None
        
        best_match = None
        best_similarity = -1
        
        for student in self.embeddings:
            similarity = float(np.dot(query_embedding, student['embedding']))
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match = {
                    'id': student['id'],
                    'roll_no': student['roll_no'],
                    'name': student['name'],
                    'similarity': similarity
                }
        
        return best_match


class AttendanceTracker:
    """Track marked attendance with cooldown"""
    def __init__(self, cooldown_seconds: int = 5):
        self.attendance_records = {}
        self.cooldown_seconds = cooldown_seconds
    
    def can_mark(self, student_id: int) -> bool:
        """Check if student can be marked (cooldown expired)"""
        if student_id not in self.attendance_records:
            return True
        
        last_marked = self.attendance_records[student_id]['timestamp']
        elapsed = time.time() - last_marked
        return elapsed >= self.cooldown_seconds
    
    def mark(self, student_id: int, match: Dict):
        """Mark student attendance"""
        self.attendance_records[student_id] = {
            'timestamp': time.time(),
            'match': match
        }


def initialize_camera(camera_index: int = 0) -> cv2.VideoCapture:
    """Initialize camera with Windows-specific settings"""
    print(f"🎥 Initializing webcam (index {camera_index}) with DirectShow backend...")
    
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        if camera_index == 0:
            print("⚠️ Camera index 0 failed, trying index 1...")
            cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
            if not cap.isOpened():
                raise RuntimeError("Failed to open camera on index 0 and 1")
        else:
            raise RuntimeError(f"Failed to open camera on index {camera_index}")
    
    print("🔧 Setting MJPG codec...")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("🔥 Warming up camera...")
    for i in range(10):
        ret, frame = cap.read()
        time.sleep(0.05)
    
    ret, frame = cap.read()
    if not ret or frame is None:
        raise RuntimeError("Camera warmup failed - cannot read frames")
    
    print(f"✅ Camera initialized successfully (shape: {frame.shape}, dtype: {frame.dtype})")
    return cap


def play_beep():
    """Play beep sound"""
    try:
        winsound.Beep(1000, 200)
    except:
        pass


def mark_attendance_in_db(db: Database, class_id: int, student_id: int, roll_no: str, name: str) -> bool:
    """Mark attendance in database - simplified version"""
    try:
        # Simply mark attendance - database will handle duplicates
        db.mark_attendance(class_id, student_id)
        return True
    except Exception as e:
        print(f"Error marking attendance: {e}")
        import traceback
        traceback.print_exc()
        return False



def draw_bbox_with_label(frame, bbox, label, color, similarity=None):
    """Draw bounding box with label"""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    label_text = label
    if similarity is not None:
        label_text = f"{label} ({similarity:.2f})"
    
    (text_width, text_height), _ = cv2.getTextSize(
        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )
    
    cv2.rectangle(
        frame,
        (x1, y1 - text_height - 10),
        (x1 + text_width, y1),
        color,
        -1
    )
    
    cv2.putText(
        frame,
        label_text,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


def run_automatic_attendance_streamlit_v2(
    class_id: int,
    db: Database,
    mtcnn,
    resnet,
    device,
    threshold: float = 0.6,
    cooldown_seconds: int = 5
) -> List[Dict]:
    """
    Run automatic attendance with Streamlit native video display
    Now with live updates and stop button
    """
    
    # Initialize stop flag in session state
    if 'stop_attendance' not in st.session_state:
        st.session_state.stop_attendance = False
    
    # Load YOLO model
    print("📦 Loading YOLO model...")
    yolo_model = YOLO("yolov8n.pt")
    print("✅ YOLO model loaded successfully")
    
    # Initialize embedding cache
    print("💾 Initializing student embedding cache...")
    embedding_cache = EmbeddingCache(db, class_id)
    print(f"✅ Loaded {len(embedding_cache.embeddings)} student(s) into cache")
    
    # Initialize tracker
    print("📊 Initializing attendance tracker...")
    tracker = AttendanceTracker(cooldown_seconds=cooldown_seconds)
    print(f"✅ Tracker initialized (cooldown: {cooldown_seconds}s)")
    
    # Initialize camera
    try:
        cap = initialize_camera(camera_index=0)
    except RuntimeError as e:
        error_msg = f"❌ Camera initialization failed: {str(e)}"
        print(error_msg)
        st.error(error_msg)
        return []
    
    # Create layout with video and table side by side
    st.success("🎥 Automatic attendance started!")
    st.info("📌 **Session will run for 30 seconds. Navigate away from this page to stop early.**")
    
    # Create two columns: video on left, table on right
    col_video, col_table = st.columns([2, 1])
    
    with col_video:
        st.markdown("#### 📹 Live Video Feed")
        video_placeholder = st.empty()
        status_placeholder = st.empty()
    
    with col_table:
        st.markdown("#### 📋 Marked Students")
        table_placeholder = st.empty()
        count_placeholder = st.empty()
    
    # Processing variables
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    frame_count = 0
    max_frames = 900  # Run for 30 seconds at 30fps
    
    # Store last detection results to prevent blinking
    last_detections = []
    
    try:
        while frame_count < max_frames:
            ret, frame = cap.read()
            
            if not ret or frame is None:
                continue
            
            frame_count += 1
            
            # Create display frame
            display_frame = frame.copy()
            
            # Run YOLO detection every 3 frames (more responsive)
            if frame_count % 3 == 1:
                results = yolo_model(frame, conf=0.4, verbose=False)
                
                # Clear and update last detections
                last_detections = []
                
                # Process detections
                for result in results:
                    boxes = result.boxes
                    
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        
                        if cls == 0:  # Person detected
                            # Crop face region
                            h, w = frame.shape[:2]
                            x1_crop = max(0, int(x1))
                            y1_crop = max(0, int(y1))
                            x2_crop = min(w, int(x2))
                            y2_crop = min(h, int(y2))
                            
                            face_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
                            
                            if face_crop.size == 0:
                                continue
                            
                            # Convert to RGB for FaceNet
                            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                            
                            # Extract embedding
                            success, embedding, msg = extract_embedding(
                                face_rgb, mtcnn, resnet, device
                            )
                            
                            detection_info = {
                                'bbox': [x1, y1, x2, y2],
                                'label': 'No Face',
                                'color': (0, 255, 255),  # Yellow
                                'similarity': None
                            }
                            
                            if success:
                                # Find match
                                match = embedding_cache.find_match(embedding, threshold)
                                
                                if match:
                                    student_id = match['id']
                                    roll_no = match['roll_no']
                                    name = match['name']
                                    similarity = match['similarity']
                                    
                                    # Check if can mark
                                    if tracker.can_mark(student_id):
                                        db_success = mark_attendance_in_db(
                                            db, class_id, student_id, roll_no, name
                                        )
                                        
                                        if db_success:
                                            tracker.mark(student_id, match)
                                            play_beep()
                                            print(f"✅ ATTENDANCE MARKED: {roll_no} ({name})")
                                            
                                            # Update table immediately (no export button during session)
                                            with col_table:
                                                update_attendance_table_simple(tracker, table_placeholder, count_placeholder)
                                    
                                    # Update detection info
                                    detection_info['label'] = f"{name} ({roll_no})"
                                    detection_info['color'] = (0, 255, 0)  # Green
                                    detection_info['similarity'] = similarity
                                else:
                                    # Unknown
                                    detection_info['label'] = "Unknown"
                                    detection_info['color'] = (0, 0, 255)  # Red
                            
                            # Store detection
                            last_detections.append(detection_info)
            
            # Draw all last detections (this prevents blinking)
            for detection in last_detections:
                draw_bbox_with_label(
                    display_frame,
                    detection['bbox'],
                    detection['label'],
                    detection['color'],
                    detection['similarity']
                )
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter >= 10:
                fps_end_time = time.time()
                fps = fps_counter / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
                fps_counter = 0
            
            # Add FPS and marked count
            info_text = f"FPS: {fps:.1f} | Marked: {len(tracker.attendance_records)}"
            cv2.putText(
                display_frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Convert BGR to RGB for Streamlit
            display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            # Display in Streamlit
            video_placeholder.image(display_frame_rgb, channels="RGB", width="stretch")
            
            # Update status
            elapsed = frame_count / 30  # Approximate seconds elapsed
            remaining = max(0, 30 - elapsed)
            status_placeholder.info(f"📊 FPS: {fps:.1f} | Marked: {len(tracker.attendance_records)} students | Time remaining: {remaining:.0f}s")
            
            # Small delay
            time.sleep(0.033)  # ~30fps
    
    except Exception as e:
        print(f"Error during attendance processing: {str(e)}")
        import traceback
        traceback.print_exc()
        st.error(f"Error: {str(e)}")
    
    finally:
        if cap is not None:
            cap.release()
            print("Camera released")
        video_placeholder.empty()
        status_placeholder.empty()
    
    # Return marked students
    marked_students = []
    for student_id, record in tracker.attendance_records.items():
        match = record['match']
        marked_students.append({
            'id': match['id'],
            'roll_no': match['roll_no'],
            'name': match['name'],
            'similarity': match['similarity'],
            'timestamp': datetime.fromtimestamp(record['timestamp']).strftime("%H:%M:%S")
        })
    
    print(f"\n📊 Total students marked: {len(marked_students)}")
    return marked_students


def update_attendance_table(tracker: AttendanceTracker, table_placeholder, count_placeholder, export_placeholder=None):
    """Update the attendance table in real-time"""
    import pandas as pd
    from datetime import datetime
    import os
    
    if len(tracker.attendance_records) > 0:
        # Create DataFrame
        df_data = []
        for idx, (student_id, record) in enumerate(tracker.attendance_records.items(), 1):
            match = record['match']
            df_data.append({
                'No.': idx,
                'Name': match['name'],
                'Roll No': match['roll_no'],
                'Time': datetime.fromtimestamp(record['timestamp']).strftime("%H:%M:%S"),
                'Similarity': f"{match['similarity']:.3f}"
            })
        
        df = pd.DataFrame(df_data)
        table_placeholder.dataframe(df, width="stretch", hide_index=True)
        count_placeholder.success(f"✅ {len(tracker.attendance_records)} student(s) marked")
        
        # Add export button below the table if placeholder provided
        if export_placeholder is not None:
            with export_placeholder.container():
                if st.button("📥 Export to Excel", width="stretch", type="secondary", key=f"export_live_{len(tracker.attendance_records)}"):
                    # Create exports directory if it doesn't exist
                    os.makedirs("exports", exist_ok=True)
                    
                    # Generate filename with class name from session state
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    class_name = st.session_state.get('selected_class', 'Unknown')
                    filename = f"exports/attendance_{class_name}_{timestamp}.xlsx"
                    
                    # Export to Excel
                    df.to_excel(filename, index=False, engine='openpyxl')
                    
                    st.success(f"✅ Exported to {filename}")
                    
                    # Provide download button
                    with open(filename, 'rb') as f:
                        st.download_button(
                            label="💾 Download Excel File",
                            data=f,
                            file_name=os.path.basename(filename),
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            width="stretch",
                            key=f"download_{timestamp}"
                        )
    else:
        table_placeholder.info("No students marked yet...")
        count_placeholder.empty()
        if export_placeholder is not None:
            export_placeholder.empty()


def update_attendance_table_simple(tracker: AttendanceTracker, table_placeholder, count_placeholder):
    """Update the attendance table without export button (used during live session)"""
    import pandas as pd
    from datetime import datetime
    
    if len(tracker.attendance_records) > 0:
        # Create DataFrame
        df_data = []
        for idx, (student_id, record) in enumerate(tracker.attendance_records.items(), 1):
            match = record['match']
            df_data.append({
                'No.': idx,
                'Name': match['name'],
                'Roll No': match['roll_no'],
                'Time': datetime.fromtimestamp(record['timestamp']).strftime("%H:%M:%S"),
                'Similarity': f"{match['similarity']:.3f}"
            })
        
        df = pd.DataFrame(df_data)
        table_placeholder.dataframe(df, width="stretch", hide_index=True)
        count_placeholder.success(f"✅ {len(tracker.attendance_records)} student(s) marked")
    else:
        table_placeholder.info("No students marked yet...")
        count_placeholder.empty()
