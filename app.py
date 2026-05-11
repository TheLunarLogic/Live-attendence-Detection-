"""
Face Recognition Attendance System - Main Streamlit Application
Using FaceNet-PyTorch (MTCNN + InceptionResnetV1)

Pages:
  1. Class Selection
  2. Manual Student Registration (webcam)
  3. Attendance (YOLO real-time)
  4. [NEW] Dataset Registration (bulk OCR-based)
  5. [NEW] Video Attendance (mp4 / avi upload)
"""

import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime
import os
import tempfile
import shutil

# ── Core modules ──────────────────────────────────────────────────────────────
from database import Database
from face_utils import (
    load_face_models,
    extract_multi_sample_embedding,
    detect_faces,
    pil_to_numpy,
    validate_image,
)
from attendance import (
    match_faces,
    generate_attendance_report,
    export_to_excel,
    calculate_attendance_statistics,
)
from auto_attendance_v2 import run_automatic_attendance_streamlit_v2

# ── New feature modules ───────────────────────────────────────────────────────
from dataset_registration import (
    register_from_dataset,
    extract_zip_to_temp,
    collect_images_from_folder,
)
from video_attendance import (
    process_video_for_attendance,
    save_uploaded_video,
)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Face Recognition Attendance System",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource
def init_database():
    """Initialize and cache database connection."""
    os.makedirs("data", exist_ok=True)
    return Database()


# ── Session state ─────────────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "page": "class_selection",
        "selected_class": None,
        "class_id": None,
        "captured_samples": [],
        "auto_marked_students": [],
        "auto_mode_active": False,
        # video attendance results stored here between reruns
        "video_attendance_df": None,
        "video_stats": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def navigate_to(page: str, class_name: str = None, class_id: int = None):
    st.session_state.page = page
    if class_name is not None:
        st.session_state.selected_class = class_name
    if class_id is not None:
        st.session_state.class_id = class_id


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Class Selection
# ══════════════════════════════════════════════════════════════════════════════
def class_selection_page(db: Database):
    st.title("📚 Face Recognition Attendance System")
    st.markdown("### Welcome! Select or create a class to get started.")

    classes = db.get_all_classes()
    class_names = [c["class_name"] for c in classes]

    col1, col2 = st.columns(2)

    # ── Left: existing classes ─────────────────────────────────────────────
    with col1:
        st.subheader("📋 Existing Classes")

        if class_names:
            selected_class = st.selectbox(
                "Select a class:",
                options=[""] + class_names,
                index=0,
                key="class_selector",
            )

            if selected_class:
                class_info = db.get_class_by_name(selected_class)
                class_id = class_info["id"]
                student_count = db.get_student_count(class_id)

                st.info(f"**{selected_class}** — **{student_count}** registered student(s)")

                btn_cols = st.columns(2)

                with btn_cols[0]:
                    if st.button("📸 Take Attendance", use_container_width=True, type="primary"):
                        if student_count == 0:
                            st.error("No students registered. Add students first.")
                        else:
                            navigate_to("attendance", selected_class, class_id)
                            st.rerun()

                with btn_cols[1]:
                    if st.button("➕ Add Students", use_container_width=True):
                        navigate_to("registration", selected_class, class_id)
                        st.rerun()

                st.markdown("---")

                extra_cols = st.columns(2)
                with extra_cols[0]:
                    if st.button("📂 Register from Dataset", use_container_width=True):
                        navigate_to("dataset_registration", selected_class, class_id)
                        st.rerun()

                with extra_cols[1]:
                    if st.button("🎬 Attendance from Video", use_container_width=True):
                        if student_count == 0:
                            st.error("No students registered. Add students first.")
                        else:
                            navigate_to("video_attendance", selected_class, class_id)
                            st.rerun()
        else:
            st.info("No classes yet. Create a new class to get started.")

    # ── Right: create new class ────────────────────────────────────────────
    with col2:
        st.subheader("✨ Create New Class")

        new_class_name = st.text_input(
            "Enter class name:",
            placeholder="e.g., BCA-6C, MCA-2A",
            key="new_class_input",
        )

        create_col, dataset_col = st.columns(2)
        with create_col:
            if st.button("Create Class", use_container_width=True, type="primary"):
                if new_class_name:
                    success, message, class_id = db.create_class(new_class_name)
                    if success:
                        st.success(message)
                        st.balloons()
                        navigate_to("registration", new_class_name, class_id)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.warning("Please enter a class name.")

        with dataset_col:
            if st.button("Create + Upload Dataset", use_container_width=True):
                if new_class_name:
                    success, message, class_id = db.create_class(new_class_name)
                    if success:
                        st.success(message)
                        navigate_to("dataset_registration", new_class_name, class_id)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.warning("Please enter a class name.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Manual Student Registration
# ══════════════════════════════════════════════════════════════════════════════
def student_registration_page(db: Database, mtcnn, resnet, device):
    st.title(f"👨‍🎓 Student Registration — {st.session_state.selected_class}")

    if st.button("← Back to Class Selection"):
        st.session_state.captured_samples = []
        navigate_to("class_selection")
        st.rerun()

    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Student Information")
        roll_no = st.text_input("Roll Number:", placeholder="e.g., 101, BCA001", key="roll_input")
        name = st.text_input("Student Name:", placeholder="e.g., John Doe", key="name_input")
        st.markdown("---")
        st.subheader("📷 Capture Face Samples")
        st.info("📌 Capture **5 clear face images**. Good lighting, one face only.")

        camera_image = st.camera_input(
            "Take a picture", key=f"camera_{len(st.session_state.captured_samples)}"
        )

        if camera_image is not None:
            pil_img = Image.open(camera_image)
            img_array = pil_to_numpy(pil_img)
            is_valid, msg = validate_image(img_array)
            if is_valid:
                st.session_state.captured_samples.append(img_array)
                st.success(f"✅ Sample {len(st.session_state.captured_samples)} captured!")
                st.rerun()
            else:
                st.error(f"Invalid image: {msg}")

    with col2:
        st.subheader("📊 Captured Samples")

        if st.session_state.captured_samples:
            st.write(f"**{len(st.session_state.captured_samples)} sample(s) captured**")
            cols = st.columns(3)
            for idx, sample in enumerate(st.session_state.captured_samples):
                with cols[idx % 3]:
                    st.image(sample, caption=f"Sample {idx + 1}", use_container_width=True)

            if st.button("🗑️ Clear All Samples", type="secondary"):
                st.session_state.captured_samples = []
                st.rerun()

            st.markdown("---")

            if len(st.session_state.captured_samples) >= 5:
                if st.button("✅ Register Student", use_container_width=True, type="primary"):
                    if not roll_no or not name:
                        st.error("Please enter both roll number and name.")
                    else:
                        with st.spinner("Processing face samples..."):
                            success, embedding, msg = extract_multi_sample_embedding(
                                st.session_state.captured_samples, mtcnn, resnet, device
                            )
                            if success:
                                db_success, db_msg = db.add_student(
                                    st.session_state.class_id, roll_no, name, embedding
                                )
                                if db_success:
                                    st.success(db_msg)
                                    st.balloons()
                                    st.session_state.captured_samples = []
                                    st.rerun()
                                else:
                                    st.error(db_msg)
                            else:
                                st.error(f"Face processing failed: {msg}")
            else:
                remaining = 5 - len(st.session_state.captured_samples)
                st.warning(f"⚠️ Please capture **{remaining} more sample(s)** (min 5 required).")
        else:
            st.info("No samples yet. Use the camera above.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Automatic / Real-time Attendance (YOLO)
# ══════════════════════════════════════════════════════════════════════════════
def attendance_page(db: Database, mtcnn, resnet, device):
    st.title(f"📸 Take Attendance — {st.session_state.selected_class}")

    if st.button("← Back to Class Selection"):
        navigate_to("class_selection")
        st.rerun()

    st.markdown("---")

    students = db.get_students_by_class(st.session_state.class_id)
    if not students:
        st.warning("No students registered. Add students first.")
        return

    st.info(f"**{len(students)}** student(s) registered in this class")

    if "auto_mode_active" not in st.session_state:
        st.session_state.auto_mode_active = False

    st.subheader("🤖 Automatic Attendance Mode (YOLO + Real-Time)")
    st.write("Click below to start real-time webcam attendance using YOLO face detection.")

    col_auto1, col_auto2 = st.columns(2)
    with col_auto1:
        if st.button("🎥 Start Automatic Attendance", use_container_width=True, type="primary", key="start_auto"):
            st.session_state.auto_mode_active = True
            st.rerun()

    with col_auto2:
        if st.button("🔄 Clear Results", use_container_width=True, type="secondary", key="clear_auto"):
            st.session_state.auto_marked_students = []
            st.session_state.auto_mode_active = False
            st.rerun()

    if st.session_state.auto_mode_active:
        marked_students = run_automatic_attendance_streamlit_v2(
            class_id=st.session_state.class_id,
            db=db,
            mtcnn=mtcnn,
            resnet=resnet,
            device=device,
        )
        st.session_state.auto_marked_students = marked_students
        st.session_state.auto_mode_active = False
        if marked_students:
            st.success(f"✅ Session complete! Marked **{len(marked_students)}** student(s)")
        else:
            st.info("Session ended. No students were marked.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 [NEW] — Dataset-Based Registration
# ══════════════════════════════════════════════════════════════════════════════
def dataset_registration_page(db: Database, mtcnn, resnet, device):
    st.title(f"📂 Register Class from Dataset — {st.session_state.selected_class}")

    if st.button("← Back to Class Selection"):
        navigate_to("class_selection")
        st.rerun()

    st.markdown("---")

    st.markdown("""
    Upload a **ZIP file** containing student images.  
    Each image must contain **only the student's face** (no text).  
    The student's name and roll number must be encoded in the **filename**.

    **Supported formats:** `.jpg`, `.jpeg`, `.png`, `.bmp`
    """)

    with st.expander("📋 Expected Image Format", expanded=False):
        st.markdown("""
        **Filename Format:** `Name_RollNo.jpg`  
        *(or any recognized image extension).*

        **Examples:**
        - `Rahul_Sharma_221045.jpg`
        - `Priya_Singh_220204.png`
        - `Amit_Kumar_Singh_210112.jpeg`
        """)

    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Dataset (ZIP)")
        uploaded_zip = st.file_uploader(
            "Upload ZIP file containing student images",
            type=["zip"],
            key="dataset_zip",
        )

        if uploaded_zip is not None:
            st.info(f"📁 Uploaded: **{uploaded_zip.name}** ({uploaded_zip.size // 1024} KB)")

        if st.button("🚀 Start Registration", use_container_width=True, type="primary"):
            if uploaded_zip is None:
                st.error("Please upload a ZIP file first.")
                return

            # Extract ZIP
            with st.spinner("Extracting ZIP file..."):
                zip_bytes = uploaded_zip.read()
                try:
                    tmp_dir, image_paths = extract_zip_to_temp(zip_bytes)
                except Exception as e:
                    st.error(f"Failed to extract ZIP: {e}")
                    return

            if not image_paths:
                st.error("No images found in the ZIP file.")
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return

            st.info(f"Found **{len(image_paths)}** image(s) to process.")

            # Progress UI
            progress_bar = st.progress(0)
            status_text = st.empty()

            def on_progress(current, total, message, *_):
                progress_bar.progress(current / total)
                status_text.text(message)

            # Run registration
            with st.spinner("Reading student details from filenames and processing images..."):
                stats = register_from_dataset(
                    image_paths=image_paths,
                    class_id=st.session_state.class_id,
                    mtcnn=mtcnn,
                    resnet=resnet,
                    device=device,
                    db=db,
                    progress_callback=on_progress,
                )

            # Cleanup temp dir
            shutil.rmtree(tmp_dir, ignore_errors=True)
            progress_bar.progress(1.0)
            status_text.empty()

            # Store stats in session state for display
            st.session_state["dataset_stats"] = stats

    # ── Right column: results summary ─────────────────────────────────────
    with col2:
        st.subheader("📊 Registration Summary")

        if "dataset_stats" in st.session_state:
            s = st.session_state["dataset_stats"]

            metric_cols = st.columns(2)
            with metric_cols[0]:
                st.metric("Total Images", s["total"])
                st.metric("✅ Registered", s["registered"])
            with metric_cols[1]:
                st.metric("⚠️ Skipped", s["skipped"])
                st.metric("❌ Filename Parsing Errors", s["parsing_failures"])

            if s["registered"] > 0:
                st.success(f"🎉 Successfully registered **{s['registered']}** student(s)!")

            if s["errors"]:
                with st.expander(f"⚠️ Issues ({len(s['errors'])} items)", expanded=False):
                    for fname, reason in s["errors"]:
                        st.text(f"• {fname}: {reason}")
        else:
            st.info("Upload a dataset ZIP and click 'Start Registration' to see results.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 [NEW] — Video-Based Attendance
# ══════════════════════════════════════════════════════════════════════════════
def video_attendance_page(db: Database, mtcnn, resnet, device):
    st.title(f"🎬 Attendance from Video — {st.session_state.selected_class}")

    if st.button("← Back to Class Selection"):
        st.session_state.video_attendance_df = None
        st.session_state.video_stats = None
        navigate_to("class_selection")
        st.rerun()

    st.markdown("---")

    students = db.get_students_by_class(st.session_state.class_id)
    if not students:
        st.warning("No students registered. Add students first.")
        return

    st.info(f"**{len(students)}** student(s) registered in this class")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🎥 Upload Classroom Video")
        st.markdown("""
        Upload a classroom recording and the system will **automatically**:
        - 🔍 Scan every frame for maximum student detection
        - 🧠 Use multi-frame verification for high accuracy
        - 📊 Mark attendance only for confidently recognized students

        > *System automatically optimizes detection for best accuracy — no manual tuning required.*
        """)

        uploaded_video = st.file_uploader(
            "Upload video file",
            type=["mp4", "avi", "mov", "mkv"],
            key="attendance_video",
        )

        if uploaded_video is not None:
            file_size_mb = uploaded_video.size / (1024 * 1024)
            st.info(f"📁 **{uploaded_video.name}** — {file_size_mb:.1f} MB")

        if st.button("▶️ Process Video", use_container_width=True, type="primary"):
            if uploaded_video is None:
                st.error("Please upload a video file.")
                return

            # Save uploaded video to temp file
            with st.spinner("Saving video..."):
                tmp_video_path = save_uploaded_video(uploaded_video)

            # Progress UI
            progress_bar = st.progress(0)
            status_text = st.empty()
            recognized_count = st.empty()

            def on_progress(frame_idx, total_frames, message, n_recognized):
                if total_frames > 0:
                    progress_bar.progress(min(frame_idx / total_frames, 1.0))
                status_text.text(message)
                recognized_count.info(f"👥 Students recognized so far: **{n_recognized}**")

            try:
                with st.spinner("Processing video for maximum student coverage..."):
                    matched_students, video_stats = process_video_for_attendance(
                        video_path=tmp_video_path,
                        students=students,
                        mtcnn=mtcnn,
                        resnet=resnet,
                        device=device,
                        progress_callback=on_progress,
                    )

                # Generate attendance report
                current_date = datetime.now().strftime("%Y-%m-%d")
                df = generate_attendance_report(matched_students, students, current_date)

                st.session_state.video_attendance_df = df
                st.session_state.video_stats = video_stats
                st.session_state.video_matched_students = matched_students

            except Exception as e:
                st.error(f"Error processing video: {e}")
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_video_path)
                except Exception:
                    pass
                progress_bar.progress(1.0)
                status_text.empty()
                recognized_count.empty()

            st.rerun()

    # ── Right column: attendance report ───────────────────────────────────
    with col2:
        st.subheader("📊 Attendance Report")

        if st.session_state.video_attendance_df is not None:
            df = st.session_state.video_attendance_df
            vs = st.session_state.video_stats or {}

            # Video stats
            with st.expander("📹 Video Processing Stats", expanded=False):
                stat_cols = st.columns(3)
                with stat_cols[0]:
                    st.metric("Total Frames", vs.get("total_frames", 0))
                with stat_cols[1]:
                    st.metric("Frames Analyzed", vs.get("processed_frames", 0))
                with stat_cols[2]:
                    st.metric("FPS", f"{vs.get('fps', 0):.1f}")

            # ── Prominent detection summary ──
            stats = calculate_attendance_statistics(df)
            pct = stats['attendance_percentage']
            st.success(
                f"**Detected: {stats['present']} / {stats['total_students']} students "
                f"({pct:.0f}%)**"
            )
            a_cols = st.columns(3)
            with a_cols[0]:
                st.metric("Total Students", stats["total_students"])
            with a_cols[1]:
                st.metric("Present ✅", stats["present"])
            with a_cols[2]:
                st.metric("Absent ❌", stats["absent"])

            st.metric("Attendance %", f"{stats['attendance_percentage']:.1f}%")
            st.markdown("---")

            # Attendance table
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Export
            if st.button("📥 Export to Excel", use_container_width=True, type="secondary"):
                success, result = export_to_excel(
                    df,
                    st.session_state.selected_class,
                    datetime.now().strftime("%Y-%m-%d"),
                )
                if success:
                    with open(result, "rb") as f:
                        st.download_button(
                            label="💾 Download Excel File",
                            data=f,
                            file_name=os.path.basename(result),
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                        )
                    st.success("✅ Exported successfully!")
                else:
                    st.error(result)
        else:
            st.info("Upload a video and click '▶️ Process Video' to generate the attendance report.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    init_session_state()
    db = init_database()

    # Load face models once
    with st.spinner("Loading FaceNet-PyTorch models..."):
        mtcnn, resnet, device = load_face_models()

    if mtcnn is None or resnet is None:
        st.error("❌ Failed to load face recognition models. Check your installation.")
        st.info("Run: `pip install facenet-pytorch torch`")
        st.stop()

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🎓 Navigation")
        st.markdown("---")

        if st.session_state.selected_class:
            st.info(f"**Current Class:**\n{st.session_state.selected_class}")
            st.markdown("---")

        st.markdown("### 🔧 System Info")
        st.markdown(f"""
- **Model**: FaceNet (VGGFace2)
- **Detector**: MTCNN + Grid Search
- **Embedding**: 512-d
- **Device**: `{str(device).upper()}`
- **Mode**: Auto-optimized
        """)

        st.markdown("---")
        st.markdown("### 🗺️ Pages")
        page_map = {
            "class_selection":    "🏠 Class Selection",
            "registration":       "👨‍🎓 Manual Registration",
            "attendance":         "📸 Real-Time Attendance",
            "dataset_registration": "📂 Dataset Registration ✨",
            "video_attendance":   "🎬 Video Attendance ✨",
        }
        current = st.session_state.page
        for key, label in page_map.items():
            prefix = "▶ " if key == current else "   "
            st.markdown(f"{prefix}{label}")

    # ── Router ────────────────────────────────────────────────────────────
    page = st.session_state.page
    if page == "class_selection":
        class_selection_page(db)
    elif page == "registration":
        student_registration_page(db, mtcnn, resnet, device)
    elif page == "attendance":
        attendance_page(db, mtcnn, resnet, device)
    elif page == "dataset_registration":
        dataset_registration_page(db, mtcnn, resnet, device)
    elif page == "video_attendance":
        video_attendance_page(db, mtcnn, resnet, device)
    else:
        class_selection_page(db)


if __name__ == "__main__":
    main()
