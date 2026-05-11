"""
Database module for Face Recognition Attendance System.
Handles SQLite database operations for classes and students.
"""

import sqlite3
import numpy as np
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime


class Database:
    """Manages SQLite database operations for the attendance system."""
    
    def __init__(self, db_path: str = "data/attendance.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.init_db()
    
    def get_connection(self) -> sqlite3.Connection:
        """Create and return a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
    
    def init_db(self):
        """Initialize database schema with classes and students tables."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Classes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS classes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                class_name TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Students table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                class_id INTEGER NOT NULL,
                roll_no TEXT NOT NULL,
                name TEXT NOT NULL,
                embedding TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (class_id) REFERENCES classes (id),
                UNIQUE(class_id, roll_no)
            )
        """)
        
        # Attendance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                class_id INTEGER NOT NULL,
                student_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                status TEXT DEFAULT 'present',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (class_id) REFERENCES classes (id),
                FOREIGN KEY (student_id) REFERENCES students (id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def create_class(self, class_name: str) -> Tuple[bool, str, Optional[int]]:
        """
        Create a new class.
        
        Args:
            class_name: Name of the class (e.g., BCA-6C)
        
        Returns:
            Tuple of (success, message, class_id)
        """
        if not class_name or not class_name.strip():
            return False, "Class name cannot be empty", None
        
        class_name = class_name.strip()
        
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("INSERT INTO classes (class_name) VALUES (?)", (class_name,))
            class_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return True, f"Class '{class_name}' created successfully", class_id
        except sqlite3.IntegrityError:
            return False, f"Class '{class_name}' already exists", None
        except Exception as e:
            return False, f"Error creating class: {str(e)}", None
    
    def get_all_classes(self) -> List[Dict]:
        """
        Retrieve all classes.
        
        Returns:
            List of dictionaries containing class information
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, class_name, created_at FROM classes ORDER BY class_name")
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_class_by_name(self, class_name: str) -> Optional[Dict]:
        """
        Get class information by name.
        
        Args:
            class_name: Name of the class
        
        Returns:
            Dictionary with class info or None if not found
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, class_name, created_at FROM classes WHERE class_name = ?", 
                      (class_name,))
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else None
    
    def check_duplicate_roll(self, class_id: int, roll_no: str) -> bool:
        """
        Check if roll number already exists in the class.
        
        Args:
            class_id: ID of the class
            roll_no: Roll number to check
        
        Returns:
            True if duplicate exists, False otherwise
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) as count FROM students WHERE class_id = ? AND roll_no = ?",
            (class_id, roll_no)
        )
        count = cursor.fetchone()['count']
        conn.close()
        
        return count > 0
    
    def add_student(self, class_id: int, roll_no: str, name: str, 
                   embedding: np.ndarray) -> Tuple[bool, str]:
        """
        Add a new student to the database.
        
        Args:
            class_id: ID of the class
            roll_no: Student's roll number
            name: Student's name
            embedding: Face embedding (512-d numpy array)
        
        Returns:
            Tuple of (success, message)
        """
        # Validation
        if not roll_no or not roll_no.strip():
            return False, "Roll number cannot be empty"
        
        if not name or not name.strip():
            return False, "Name cannot be empty"
        
        roll_no = roll_no.strip()
        name = name.strip()
        
        # Check for duplicate roll number
        if self.check_duplicate_roll(class_id, roll_no):
            return False, f"Roll number '{roll_no}' already exists in this class"
        
        # Validate embedding
        if embedding is None or embedding.size == 0:
            return False, "Invalid face embedding"
        
        try:
            # Serialize embedding to JSON
            embedding_json = self._serialize_embedding(embedding)
            
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO students (class_id, roll_no, name, embedding) VALUES (?, ?, ?, ?)",
                (class_id, roll_no, name, embedding_json)
            )
            conn.commit()
            conn.close()
            
            return True, f"Student '{name}' (Roll: {roll_no}) added successfully"
        except Exception as e:
            return False, f"Error adding student: {str(e)}"
    
    def get_students_by_class(self, class_id: int) -> List[Dict]:
        """
        Retrieve all students for a specific class.
        
        Args:
            class_id: ID of the class
        
        Returns:
            List of dictionaries containing student information with embeddings
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, roll_no, name, embedding FROM students WHERE class_id = ? ORDER BY roll_no",
            (class_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        
        students = []
        for row in rows:
            student = dict(row)
            # Deserialize embedding
            student['embedding'] = self._deserialize_embedding(student['embedding'])
            students.append(student)
        
        return students
    
    def get_student_count(self, class_id: int) -> int:
        """
        Get the number of students in a class.
        
        Args:
            class_id: ID of the class
        
        Returns:
            Number of students
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM students WHERE class_id = ?", (class_id,))
        count = cursor.fetchone()['count']
        conn.close()
        
        return count
    
    def mark_attendance(self, class_id: int, student_id: int) -> bool:
        """
        Mark attendance for a student.
        
        Args:
            class_id: ID of the class
            student_id: ID of the student
        
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Get current date and time
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")
            
            # Check if already marked today
            cursor.execute("""
                SELECT id FROM attendance 
                WHERE class_id = ? AND student_id = ? AND date = ?
            """, (class_id, student_id, date_str))
            
            if cursor.fetchone():
                # Already marked today
                conn.close()
                return False
            
            # Mark attendance
            cursor.execute("""
                INSERT INTO attendance (class_id, student_id, date, time, status)
                VALUES (?, ?, ?, ?, 'present')
            """, (class_id, student_id, date_str, time_str))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error marking attendance: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_attendance_by_date_range(self, class_id: int, start_date: str, end_date: str) -> List[Dict]:
        """
        Get attendance records for a date range.
        
        Args:
            class_id: ID of the class
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            List of attendance records
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT a.id, a.student_id, s.roll_no, s.name, a.date, a.time, a.status
                FROM attendance a
                JOIN students s ON a.student_id = s.id
                WHERE a.class_id = ? AND a.date BETWEEN ? AND ?
                ORDER BY a.date DESC, a.time DESC
            """, (class_id, start_date, end_date))
            
            records = []
            for row in cursor.fetchall():
                records.append({
                    'id': row[0],
                    'student_id': row[1],
                    'roll_no': row[2],
                    'name': row[3],
                    'date': row[4],
                    'time': row[5],
                    'status': row[6]
                })
            
            conn.close()
            return records
            
        except Exception as e:
            print(f"Error getting attendance: {e}")
            return []
    
    def _serialize_embedding(self, embedding: np.ndarray) -> str:
        """
        Serialize numpy array to JSON string.
        
        Args:
            embedding: Numpy array
        
        Returns:
            JSON string
        """
        return json.dumps(embedding.tolist())
    
    def _deserialize_embedding(self, json_str: str) -> np.ndarray:
        """
        Deserialize JSON string back to numpy array.
        
        Args:
            json_str: JSON string
        
        Returns:
            Numpy array
        """
        return np.array(json.loads(json_str), dtype=np.float32)
