import sqlite3
import datetime
import os
import logging

class Database:
    def __init__(self, db_path="data/watchdog.db"):
        self.db_path = db_path
        # Ensure the data directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._create_table()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _create_table(self):
        """Creates the events table if it doesn't exist."""
        query = """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_text TEXT,
            province TEXT,
            timestamp DATETIME,
            entry_time DATETIME,
            duration REAL,
            image_path TEXT,
            is_illegal BOOLEAN
        );
        """
        with self._get_connection() as conn:
            conn.execute(query)

    def save_event(self, plate_text, province, entry_time, image_frame, is_illegal):
        """
        Saves the metadata and the actual image file.
        """
        now = datetime.datetime.now()
        duration = (now - entry_time).total_seconds()
        
        # 1. Save the image to the disk first
        # Structure: data/evidence/YYYY-MM-DD/HHMMSS_PLATE.jpg
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H%M%S")
        folder_path = os.path.join("data/evidence", date_str)
        os.makedirs(folder_path, exist_ok=True)
        
        image_filename = f"{time_str}_{plate_text}.jpg"
        image_path = os.path.join(folder_path, image_filename)
        
        import cv2
        cv2.imwrite(image_path, image_frame)

        # 2. Save metadata to SQLite
        query = """
        INSERT INTO events (plate_text, province, timestamp, entry_time, duration, image_path, is_illegal)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self._get_connection() as conn:
            conn.execute(query, (
                plate_text, 
                province, 
                now.isoformat(), 
                entry_time.isoformat(), 
                duration, 
                image_path, 
                is_illegal
            ))
        
        logging.info(f"Database: Event saved for {plate_text} at {image_path}")
        return image_path

    def get_frequent_offenders(self, limit=5):
        """Optional: Find plates that appear most often."""
        query = """
        SELECT plate_text, COUNT(*) as count 
        FROM events 
        WHERE is_illegal = 1 
        GROUP BY plate_text 
        ORDER BY count DESC 
        LIMIT ?
        """
        with self._get_connection() as conn:
            return conn.execute(query, (limit,)).fetchall()
