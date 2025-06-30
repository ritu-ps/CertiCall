# database.py
import sqlite3
from datetime import datetime

def init_db():
    conn = sqlite3.connect('meetings.db')
    c = conn.cursor()
    
    # Create tables if they don't exist
    c.execute('''CREATE TABLE IF NOT EXISTS hosts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 name TEXT NOT NULL,
                 email TEXT UNIQUE NOT NULL,
                 password TEXT NOT NULL,
                 company TEXT NOT NULL)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS meetings
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 host_id INTEGER NOT NULL,
                 title TEXT NOT NULL,
                 description TEXT,
                 start_time DATETIME NOT NULL,
                 end_time DATETIME,
                 FOREIGN KEY (host_id) REFERENCES hosts (id))''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS employees
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 meeting_id INTEGER NOT NULL,
                 name TEXT NOT NULL,
                 emp_id TEXT NOT NULL,
                 password TEXT NOT NULL,
                 FOREIGN KEY (meeting_id) REFERENCES meetings (id))''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 meeting_id INTEGER NOT NULL,
                 emp_id TEXT NOT NULL,
                 name TEXT NOT NULL,
                 gender TEXT NOT NULL,
                 join_time DATETIME NOT NULL,
                 lie_detected BOOLEAN DEFAULT FALSE,
                 lie_timestamps TEXT,
                 FOREIGN KEY (meeting_id) REFERENCES meetings (id))''')
    
    conn.commit()
    conn.close()

def add_host(name, email, password, company):
    conn = sqlite3.connect('meetings.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO hosts (name, email, password, company) VALUES (?, ?, ?, ?)",
                  (name, email, password, company))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_host(email, password):
    conn = sqlite3.connect('meetings.db')
    c = conn.cursor()
    c.execute("SELECT id, name, company FROM hosts WHERE email=? AND password=?", (email, password))
    result = c.fetchone()
    conn.close()
    return result if result else None

def create_meeting(host_id, title, description, start_time, end_time=None):
    conn = sqlite3.connect('meetings.db')
    c = conn.cursor()
    c.execute("INSERT INTO meetings (host_id, title, description, start_time, end_time) VALUES (?, ?, ?, ?, ?)",
              (host_id, title, description, start_time, end_time))
    meeting_id = c.lastrowid
    conn.commit()
    conn.close()
    return meeting_id

def add_employee(meeting_id, name, emp_id, password):
    conn = sqlite3.connect('meetings.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO employees (meeting_id, name, emp_id, password) VALUES (?, ?, ?, ?)",
                  (meeting_id, name, emp_id, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_employee(meeting_id, emp_id, password):
    conn = sqlite3.connect('meetings.db')
    c = conn.cursor()
    c.execute("SELECT name FROM employees WHERE meeting_id=? AND emp_id=? AND password=?", 
              (meeting_id, emp_id, password))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

def record_attendance(meeting_id, emp_id, name, gender, lie_detected=False, lie_timestamps=None):
    conn = sqlite3.connect('meetings.db')
    c = conn.cursor()
    c.execute("INSERT INTO attendance (meeting_id, emp_id, name, gender, join_time, lie_detected, lie_timestamps) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (meeting_id, emp_id, name, gender, datetime.now(), lie_detected, str(lie_timestamps) if lie_timestamps else None))
    conn.commit()
    conn.close()

def get_meetings_for_host(host_id):
    conn = sqlite3.connect('meetings.db')
    c = conn.cursor()
    c.execute("SELECT id, title, start_time FROM meetings WHERE host_id=?", (host_id,))
    result = c.fetchall()
    conn.close()
    return result

def get_attendance_for_meeting(meeting_id):
    conn = sqlite3.connect('meetings.db')
    c = conn.cursor()
    c.execute("SELECT emp_id, name, gender, join_time, lie_detected, lie_timestamps FROM attendance WHERE meeting_id=?", (meeting_id,))
    result = c.fetchall()
    conn.close()
    return result

def get_employees_for_meeting(meeting_id):
    conn = sqlite3.connect('meetings.db')
    c = conn.cursor()
    c.execute("SELECT emp_id, name FROM employees WHERE meeting_id=?", (meeting_id,))
    result = c.fetchall()
    conn.close()
    return result
def record_basic_attendance(meeting_id, emp_id, name, gender):
    """Record basic attendance info before video call"""
    conn = sqlite3.connect('meetings.db')
    c = conn.cursor()
    c.execute("""
        INSERT INTO attendance (meeting_id, emp_id, name, gender, join_time)
        VALUES (?, ?, ?, ?, datetime('now'))
    """, (meeting_id, emp_id, name, gender))
    conn.commit()
    conn.close()

def update_suspicious_moments(meeting_id, emp_id, suspicious_moments):
    """Update the suspicious moments after video call"""
    conn = sqlite3.connect('meetings.db')
    c = conn.cursor()
    c.execute("""
        UPDATE attendance 
        SET lie_detected=?, lie_timestamps=?
        WHERE meeting_id=? AND emp_id=?
    """, (bool(suspicious_moments), suspicious_moments, meeting_id, emp_id))
    conn.commit()
    conn.close()