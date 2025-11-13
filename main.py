import streamlit as st
import time
from datetime import datetime
import json
import os
import hashlib
import sqlite3
import random

# Page config must be the first Streamlit command
st.set_page_config(page_title="CertiCall", layout="wide", initial_sidebar_state="expanded")

# Database functions
def init_db():
    """Initialize the SQLite database"""
    try:
        conn = sqlite3.connect('meetings.db', check_same_thread=False)
        c = conn.cursor()
        
        # Create tables if they don't exist
        c.execute('''CREATE TABLE IF NOT EXISTS hosts
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      name TEXT NOT NULL,
                      email TEXT UNIQUE NOT NULL,
                      password TEXT NOT NULL,
                      company TEXT NOT NULL,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS meetings
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      host_id INTEGER,
                      title TEXT NOT NULL,
                      description TEXT,
                      start_time TIMESTAMP,
                      end_time TIMESTAMP,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      is_active BOOLEAN DEFAULT TRUE)''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS employees
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      meeting_id INTEGER,
                      name TEXT NOT NULL,
                      emp_id TEXT NOT NULL,
                      password TEXT NOT NULL,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      UNIQUE(meeting_id, emp_id))''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS attendance
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      meeting_id INTEGER,
                      emp_id TEXT NOT NULL,
                      name TEXT NOT NULL,
                      gender TEXT,
                      join_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      lie_detected BOOLEAN DEFAULT FALSE,
                      lie_timestamps TEXT,
                      meeting_duration INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS face_dataset
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      name TEXT NOT NULL,
                      meeting_id INTEGER,
                      image_path TEXT,
                      registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      features TEXT)''')
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Database error: {e}")
        return False

def hash_password(password):
    """Simple password hashing"""
    return hashlib.sha256(password.encode()).hexdigest()

def add_host(name, email, password, company):
    """Add a new host to the database"""
    try:
        conn = sqlite3.connect('meetings.db', check_same_thread=False)
        c = conn.cursor()
        hashed_pwd = hash_password(password)
        c.execute("INSERT INTO hosts (name, email, password, company) VALUES (?, ?, ?, ?)",
                 (name, email, hashed_pwd, company))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False
    except Exception as e:
        return False

def verify_host(email, password):
    """Verify host credentials"""
    try:
        conn = sqlite3.connect('meetings.db', check_same_thread=False)
        c = conn.cursor()
        hashed_pwd = hash_password(password)
        c.execute("SELECT id, name, company FROM hosts WHERE email=? AND password=?", 
                 (email, hashed_pwd))
        result = c.fetchone()
        conn.close()
        return result
    except Exception as e:
        return None

def create_meeting(host_id, title, description, start_time, end_time=None):
    """Create a new meeting"""
    try:
        conn = sqlite3.connect('meetings.db', check_same_thread=False)
        c = conn.cursor()
        c.execute("INSERT INTO meetings (host_id, title, description, start_time, end_time) VALUES (?, ?, ?, ?, ?)",
                 (host_id, title, description, start_time, end_time))
        meeting_id = c.lastrowid
        conn.commit()
        conn.close()
        return meeting_id
    except Exception as e:
        return None

def get_meetings_for_host(host_id):
    """Get all meetings for a host"""
    try:
        conn = sqlite3.connect('meetings.db', check_same_thread=False)
        c = conn.cursor()
        c.execute("SELECT id, title, start_time FROM meetings WHERE host_id=? ORDER BY start_time DESC", (host_id,))
        results = c.fetchall()
        conn.close()
        return results
    except Exception as e:
        return []

def add_employee(meeting_id, name, emp_id, password):
    """Add an employee to a meeting"""
    try:
        conn = sqlite3.connect('meetings.db', check_same_thread=False)
        c = conn.cursor()
        hashed_pwd = hash_password(password)
        c.execute("INSERT INTO employees (meeting_id, name, emp_id, password) VALUES (?, ?, ?, ?)",
                 (meeting_id, name, emp_id, hashed_pwd))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False
    except Exception as e:
        return False

def get_employees_for_meeting(meeting_id):
    """Get all employees for a meeting"""
    try:
        conn = sqlite3.connect('meetings.db', check_same_thread=False)
        c = conn.cursor()
        c.execute("SELECT emp_id, name FROM employees WHERE meeting_id=?", (meeting_id,))
        results = c.fetchall()
        conn.close()
        return results
    except Exception as e:
        return []

def verify_employee(meeting_id, emp_id, password):
    """Verify employee credentials"""
    try:
        conn = sqlite3.connect('meetings.db', check_same_thread=False)
        c = conn.cursor()
        hashed_pwd = hash_password(password)
        c.execute("SELECT name FROM employees WHERE meeting_id=? AND emp_id=? AND password=?", 
                 (meeting_id, emp_id, hashed_pwd))
        result = c.fetchone()
        conn.close()
        return result
    except Exception as e:
        return None

def record_basic_attendance(meeting_id, emp_id, name, gender):
    """Record basic attendance information"""
    try:
        conn = sqlite3.connect('meetings.db', check_same_thread=False)
        c = conn.cursor()
        c.execute("INSERT INTO attendance (meeting_id, emp_id, name, gender) VALUES (?, ?, ?, ?)",
                 (meeting_id, emp_id, name, gender))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        return False

def update_meeting_duration(meeting_id, emp_id, duration):
    """Update meeting duration"""
    try:
        conn = sqlite3.connect('meetings.db', check_same_thread=False)
        c = conn.cursor()
        c.execute('''UPDATE attendance 
                     SET meeting_duration=?
                     WHERE meeting_id=? AND emp_id=? AND id = (
                         SELECT id FROM attendance 
                         WHERE meeting_id=? AND emp_id=? 
                         ORDER BY join_time DESC LIMIT 1
                     )''',
                 (duration, meeting_id, emp_id, meeting_id, emp_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        return False

def update_suspicious_moments(meeting_id, emp_id, lie_timestamps):
    """Update suspicious moments for an attendance record"""
    try:
        conn = sqlite3.connect('meetings.db', check_same_thread=False)
        c = conn.cursor()
        c.execute('''UPDATE attendance 
                     SET lie_detected=?, lie_timestamps=?
                     WHERE meeting_id=? AND emp_id=? AND id = (
                         SELECT id FROM attendance 
                         WHERE meeting_id=? AND emp_id=? 
                         ORDER BY join_time DESC LIMIT 1
                     )''',
                 (True, lie_timestamps, meeting_id, emp_id, meeting_id, emp_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        return False

def get_attendance_for_meeting(meeting_id):
    """Get attendance records for a meeting"""
    try:
        conn = sqlite3.connect('meetings.db', check_same_thread=False)
        c = conn.cursor()
        c.execute('''SELECT emp_id, name, gender, join_time, lie_detected, lie_timestamps, meeting_duration 
                     FROM attendance WHERE meeting_id=? ORDER BY join_time DESC''',
                 (meeting_id,))
        results = c.fetchall()
        conn.close()
        return results
    except Exception as e:
        return []

def save_face_to_database(name, meeting_id, features=None):
    """Save face data to database"""
    try:
        conn = sqlite3.connect('meetings.db', check_same_thread=False)
        c = conn.cursor()
        features_json = json.dumps(features) if features else "{}"
        c.execute("INSERT INTO face_dataset (name, meeting_id, features) VALUES (?, ?, ?)",
                 (name, meeting_id, features_json))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        return False

def get_face_dataset():
    """Get all registered faces"""
    try:
        conn = sqlite3.connect('meetings.db', check_same_thread=False)
        c = conn.cursor()
        c.execute("SELECT name, meeting_id, registered_at FROM face_dataset ORDER BY registered_at DESC")
        results = c.fetchall()
        conn.close()
        return results
    except Exception as e:
        return []

# Session state initialization
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_type' not in st.session_state:
    st.session_state.user_type = None
if 'host_info' not in st.session_state:
    st.session_state.host_info = None
if 'current_meeting' not in st.session_state:
    st.session_state.current_meeting = None
if 'employee_info' not in st.session_state:
    st.session_state.employee_info = None
if 'analysis_in_progress' not in st.session_state:
    st.session_state.analysis_in_progress = False
if 'in_video_call' not in st.session_state:
    st.session_state.in_video_call = False
if 'basic_info_collected' not in st.session_state:
    st.session_state.basic_info_collected = False
if 'suspicious_moments' not in st.session_state:
    st.session_state.suspicious_moments = []
if 'unknown_face_detected' not in st.session_state:
    st.session_state.unknown_face_detected = False
if 'capture_unknown_face' not in st.session_state:
    st.session_state.capture_unknown_face = False
if 'unknown_face_name' not in st.session_state:
    st.session_state.unknown_face_name = ""
if 'meeting_start_time' not in st.session_state:
    st.session_state.meeting_start_time = None
if 'face_detection_count' not in st.session_state:
    st.session_state.face_detection_count = 0

# Initialize database on startup
init_db()

def save_unknown_face_to_dataset(person_name, meeting_id):
    """Save unknown face entry to dataset"""
    try:
        # Create face_dataset directory if it doesn't exist
        dataset_dir = "face_dataset"
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        
        # Create person-specific directory
        person_dir = os.path.join(dataset_dir, person_name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{person_name}_{meeting_id}_{timestamp}.json"
        filepath = os.path.join(person_dir, filename)
        
        # Save metadata with simulated facial features
        metadata = {
            "name": person_name,
            "meeting_id": meeting_id,
            "timestamp": timestamp,
            "registered_at": datetime.now().isoformat(),
            "type": "simulated_registration",
            "facial_features": {
                "face_encoding": [random.random() for _ in range(128)],
                "landmarks": {
                    "left_eye": [random.randint(100, 200), random.randint(100, 200)],
                    "right_eye": [random.randint(100, 200), random.randint(100, 200)],
                    "nose": [random.randint(100, 200), random.randint(100, 200)],
                    "mouth_left": [random.randint(100, 200), random.randint(100, 200)],
                    "mouth_right": [random.randint(100, 200), random.randint(100, 200)]
                }
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Also save to database
        save_face_to_database(person_name, meeting_id, metadata["facial_features"])
        
        # Also save to a general unknown faces log
        log_file = os.path.join(dataset_dir, "unknown_faces_log.csv")
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                f.write("timestamp,name,meeting_id,filepath\n")
        
        with open(log_file, 'a') as f:
            f.write(f"{timestamp},{person_name},{meeting_id},{filepath}\n")
        
        return True, filepath
    except Exception as e:
        return False, str(e)

def simulate_face_recognition():
    """Simulate face recognition process"""
    # Simulate face detection with random results
    detected_faces = random.randint(0, 2)
    confidence = random.uniform(0.7, 0.99)
    
    if detected_faces > 0:
        return {
            "faces_detected": detected_faces,
            "confidence": confidence,
            "person_identified": random.choice([True, False]),
            "match_quality": random.choice(["High", "Medium", "Low"])
        }
    else:
        return {
            "faces_detected": 0,
            "confidence": 0.0,
            "person_identified": False,
            "match_quality": "None"
        }

def capture_unknown_face_interface():
    """Interface for capturing and registering unknown faces"""
    st.warning("🔍 Unknown Face Detected!")
    
    st.subheader("Register New Person")
    person_name = st.text_input("Enter person's name:", 
                               value=st.session_state.unknown_face_name,
                               key="unknown_face_name_input")
    
    # Show simulated face preview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(45deg, #667eea, #764ba2); padding: 20px; border-radius: 10px; text-align: center; color: white;'>
            <h3>📷 Face Preview</h3>
            <p>Simulated face capture</p>
            <div style='background: rgba(255,255,255,0.2); padding: 10px; border-radius: 5px; margin: 10px 0;'>
                <p>🎭 Face detected</p>
                <p>📐 128 facial features</p>
                <p>🎯 Ready for registration</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("💾 Save Face to Dataset", type="primary", use_container_width=True):
            if person_name.strip():
                emp = st.session_state.employee_info
                success, result = save_unknown_face_to_dataset(
                    person_name.strip(),
                    emp['meeting_id']
                )
                
                if success:
                    st.success(f"✅ Face registered for {person_name}!")
                    st.info(f"✅ Saved to database and file system")
                    
                    # Update the current session with the new name
                    st.session_state.employee_info['detected_name'] = person_name.strip()
                    st.session_state.unknown_face_detected = False
                    st.session_state.capture_unknown_face = False
                    st.session_state.unknown_face_name = ""
                    
                    # Add to suspicious moments log
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    st.session_state.suspicious_moments.append(
                        (timestamp, f"New face registered: {person_name}")
                    )
                    
                    st.rerun()
                else:
                    st.error(f"Failed to save face: {result}")
            else:
                st.error("Please enter a valid name")
        
        if st.button("⏭️ Skip Registration", use_container_width=True):
            st.session_state.unknown_face_detected = False
            st.session_state.capture_unknown_face = False
            st.session_state.unknown_face_name = ""
            st.rerun()

def show_login_page():
    """Show login options for both host and employee"""
    st.markdown("""
    <div style='background: linear-gradient(45deg, #667eea, #764ba2); padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;'>
        <h1 style='text-align: center; margin: 0;'>🎯 CertiCall</h1>
        <p style='text-align: center; margin: 5px 0 0 0;'>Secure Meeting Authentication & Monitoring</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("🌐 **Cloud Mode**: All advanced features are simulated for demonstration")
    
    tab1, tab2 = st.tabs(["🏢 Host Portal", "👤 Employee Portal"])
    
    with tab1:
        st.header("Host Authentication")
        login_tab, register_tab = st.tabs(["🔐 Login", "📝 Register"])
        
        with login_tab:
            st.subheader("Host Login")
            email = st.text_input("Email", key="host_login_email")
            password = st.text_input("Password", type="password", key="host_login_password")
            
            if st.button("Login as Host", key="host_login_button", type="primary", use_container_width=True):
                if not email or not password:
                    st.error("Please fill all fields")
                else:
                    with st.spinner("Authenticating..."):
                        time.sleep(1)
                        host_info = verify_host(email, password)
                        if host_info:
                            st.session_state.logged_in = True
                            st.session_state.user_type = 'host'
                            st.session_state.host_info = {
                                "id": host_info[0],
                                "name": host_info[1],
                                "company": host_info[2]
                            }
                            st.rerun()
                        else:
                            st.error("Invalid email or password")
        
        with register_tab:
            st.subheader("Host Registration")
            name = st.text_input("Full Name", key="host_reg_name")
            company = st.text_input("Company Name", key="host_reg_company")
            email = st.text_input("Email", key="host_reg_email")
            password = st.text_input("Password", type="password", key="host_reg_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="host_reg_confirm_password")
            
            if st.button("Register as Host", key="host_register_button", type="primary", use_container_width=True):
                if password != confirm_password:
                    st.error("Passwords do not match")
                elif not all([name, company, email, password]):
                    st.error("Please fill all fields")
                else:
                    with st.spinner("Creating account..."):
                        time.sleep(1)
                        if add_host(name, email, password, company):
                            st.success("Registration successful! Please login.")
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error("Email already registered")

    with tab2:
        st.header("Employee Login")
        meeting_id = st.text_input("Meeting ID", key="emp_meeting_id")
        emp_id = st.text_input("Employee ID", key="emp_id")
        password = st.text_input("Password", type="password", key="emp_password")
        
        if st.button("Join Meeting", key="emp_login_button", type="primary", use_container_width=True):
            if not all([meeting_id, emp_id, password]):
                st.error("Please fill all fields")
            else:
                with st.spinner("Verifying credentials..."):
                    time.sleep(1)
                    employee_info = verify_employee(meeting_id, emp_id, password)
                    if employee_info:
                        st.session_state.logged_in = True
                        st.session_state.user_type = 'employee'
                        st.session_state.employee_info = {
                            "meeting_id": meeting_id,
                            "emp_id": emp_id,
                            "name": employee_info[0]
                        }
                        st.rerun()
                    else:
                        st.error("Invalid credentials or meeting ID")

def host_dashboard():
    """Host dashboard after login"""
    host = st.session_state.host_info
    
    # Sidebar
    with st.sidebar:
        st.title(f"🏢 {host['company']}")
        st.subheader(f"Welcome, {host['name']}")
        st.markdown("---")
        
        st.metric("Total Meetings", len(get_meetings_for_host(host['id'])))
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_type = None
            st.session_state.host_info = None
            st.rerun()
    
    tab1, tab2, tab3, tab4 = st.tabs(["📅 Create Meeting", "👥 Manage Employees", "📊 View Attendance", "📁 Face Database"])
    
    with tab1:
        st.header("Create New Meeting")
        col1, col2 = st.columns(2)
        with col1:
            title = st.text_input("Meeting Title *", key="meeting_title", placeholder="Enter meeting title")
            start_time = st.date_input("Date *", key="meeting_date")
        with col2:
            description = st.text_area("Description", key="meeting_description", placeholder="Meeting description (optional)")
            start_hour = st.time_input("Start Time *", key="meeting_start_time")
        
        end_hour = st.time_input("End Time (optional)", value=None, key="meeting_end_time")
        
        if st.button("Create Meeting", key="create_meeting_button", type="primary", use_container_width=True):
            if not title:
                st.error("Please enter a meeting title")
            else:
                start_datetime = datetime.combine(start_time, start_hour)
                end_datetime = datetime.combine(start_time, end_hour) if end_hour else None
                meeting_id = create_meeting(host['id'], title, description, start_datetime, end_datetime)
                
                if meeting_id:
                    st.session_state.current_meeting = meeting_id
                    st.success(f"✅ Meeting '{title}' created successfully!")
                    
                    # Display sharing options
                    st.subheader("Share Meeting Access")
                    st.markdown("**Share this Meeting ID with participants:**")
                    st.code(f"Meeting ID: {meeting_id}", language="text")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"**Meeting ID:** `{meeting_id}`")
                    with col2:
                        st.info(f"**Start Time:** {start_datetime.strftime('%Y-%m-%d %H:%M')}")
                else:
                    st.error("Failed to create meeting")
    
    with tab2:
        st.header("Manage Employees")
        meetings = get_meetings_for_host(host['id'])
        if not meetings:
            st.warning("No meetings found. Please create a meeting first.")
        else:
            meeting_options = {f"{m[1]} (ID: {m[0]}) - {m[2].split()[0]}": m[0] for m in meetings}
            selected_meeting = st.selectbox(
                "Select Meeting", 
                options=list(meeting_options.keys()),
                key="manage_employees_select_meeting"
            )
            meeting_id = meeting_options[selected_meeting]
            
            st.subheader("Add New Employee")
            col1, col2, col3 = st.columns(3)
            with col1:
                emp_name = st.text_input("Employee Name *", key="add_emp_name")
            with col2:
                emp_id = st.text_input("Employee ID *", key="add_emp_id")
            with col3:
                emp_password = st.text_input("Password *", type="password", key="add_emp_password")
            
            if st.button("Add Employee", key="add_employee_button", type="primary", use_container_width=True):
                if not all([emp_name, emp_id, emp_password]):
                    st.error("Please fill all required fields")
                else:
                    if add_employee(meeting_id, emp_name, emp_id, emp_password):
                        st.success(f"✅ Employee {emp_name} added successfully!")
                    else:
                        st.error("Employee ID already exists for this meeting")
            
            st.subheader("Current Employees")
            employees = get_employees_for_meeting(meeting_id)
            if employees:
                for emp_id, name in employees:
                    with st.expander(f"👤 {name} (ID: {emp_id})"):
                        st.code(f"Meeting ID: {meeting_id}\nEmployee ID: {emp_id}\nName: {name}", language="text")
            else:
                st.info("No employees added yet for this meeting")

    with tab3:
        st.header("View Attendance")
        meetings = get_meetings_for_host(host['id'])
        if not meetings:
            st.warning("No meetings found.")
        else:
            meeting_options = {f"{m[1]} (ID: {m[0]}) - {m[2].split()[0]}": m[0] for m in meetings}
            selected_meeting = st.selectbox(
                "Select Meeting", 
                options=list(meeting_options.keys()),
                key="view_attendance_select_meeting"
            )
            meeting_id = meeting_options[selected_meeting]
            
            attendance = get_attendance_for_meeting(meeting_id)
            if attendance:
                st.subheader(f"Attendance Records ({len(attendance)} participants)")
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Participants", len(attendance))
                with col2:
                    suspicious_count = sum(1 for record in attendance if record[4])  # lie_detected field
                    st.metric("Behavior Alerts", suspicious_count)
                with col3:
                    avg_duration = sum(record[6] for record in attendance if record[6]) / len(attendance) if attendance else 0
                    st.metric("Avg Duration (min)", f"{avg_duration:.1f}")
                with col4:
                    female_count = sum(1 for record in attendance if record[2] and record[2].lower() == "female")
                    st.metric("Female Participants", female_count)
                
                for emp_id, name, gender, join_time, lie_detected, lie_timestamps, duration in attendance:
                    try:
                        if isinstance(join_time, str):
                            join_time_str = join_time
                        else:
                            join_time_str = join_time.strftime('%Y-%m-%d %H:%M:%S')
                    except Exception as e:
                        join_time_str = str(join_time)
                    
                    with st.expander(f"👤 {name} ({gender}) - {join_time_str}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Employee ID:** {emp_id}")
                            st.write(f"**Join Time:** {join_time_str}")
                            st.write(f"**Meeting Duration:** {duration} minutes")
                        with col2:
                            if lie_detected:
                                st.error("**⚠️ Behavior Alert!**")
                                if lie_timestamps:
                                    try:
                                        if isinstance(lie_timestamps, str):
                                            timestamps = json.loads(lie_timestamps.replace("'", '"'))
                                        else:
                                            timestamps = lie_timestamps
                                        st.write("**Suspicious moments:**")
                                        for ts in timestamps:
                                            st.write(f"- {ts[0]}: {ts[1]}")
                                    except:
                                        st.warning("Could not parse behavior timestamps")
                            else:
                                st.success("✅ No suspicious behavior detected")
            else:
                st.info("No attendance records yet for this meeting")

    with tab4:
        st.header("Face Database")
        face_data = get_face_dataset()
        
        if face_data:
            st.subheader(f"Registered Faces ({len(face_data)} total)")
            
            # Display face statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Faces", len(face_data))
            with col2:
                unique_names = len(set(record[0] for record in face_data))
                st.metric("Unique Persons", unique_names)
            with col3:
                recent_count = sum(1 for record in face_data if datetime.now() - datetime.strptime(record[2].split()[0], '%Y-%m-%d') < timedelta(days=7))
                st.metric("Recent (7 days)", recent_count)
            
            # Display face records
            for name, meeting_id, registered_at in face_data:
                with st.expander(f"👤 {name} - Meeting {meeting_id}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Name:** {name}")
                        st.write(f"**Meeting ID:** {meeting_id}")
                        st.write(f"**Registered:** {registered_at}")
                    with col2:
                        st.markdown("""
                        <div style='background: #f0f2f6; padding: 10px; border-radius: 5px; text-align: center;'>
                            <p>🎭 Simulated Face Data</p>
                            <p>📐 128 facial features</p>
                            <p>✅ Ready for recognition</p>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("No faces registered in the database yet.")

def employee_interface():
    """Employee interface after joining meeting"""
    if st.session_state.in_video_call:
        video_call_session()
        return
    
    emp = st.session_state.employee_info
    
    # Sidebar
    with st.sidebar:
        st.title("👤 Employee Portal")
        st.write(f"**Name:** {emp['name']}")
        st.write(f"**Employee ID:** {emp['emp_id']}")
        st.write(f"**Meeting ID:** {emp['meeting_id']}")
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_type = None
            st.session_state.employee_info = None
            st.rerun()
    
    st.title(f"🎯 Meeting Attendance Portal")
    st.subheader(f"Welcome, {emp['name']}!")
    
    st.info("""
    **Instructions for Attendance:**
    1. Click 'Begin Attendance Check' to start the face recognition process
    2. The system will simulate camera access and face detection
    3. Once verified, you'll proceed to the secure meeting session
    4. During the meeting, behavior monitoring will be active
    """)
    
    if not st.session_state.analysis_in_progress:
        if st.button("Begin Attendance Check", key="begin_attendance_check", type="primary", use_container_width=True):
            st.session_state.analysis_in_progress = True
            st.rerun()
    else:
        perform_attendance_check()

def perform_attendance_check():
    """Simulated attendance check with face recognition"""
    emp = st.session_state.employee_info
    
    st.info("🔍 **Starting Face Recognition Process...**")
    
    # Create progress visualization
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.empty()
    
    steps = [
        ("Initializing camera...", 10),
        ("Detecting face in frame...", 25),
        ("Analyzing facial features...", 45),
        ("Comparing with database...", 65),
        ("Verifying identity...", 85),
        ("Finalizing recognition...", 100)
    ]
    
    face_results = []
    
    for step, progress in steps:
        progress_bar.progress(progress)
        status_text.text(f"{step} {progress}%")
        
        # Simulate face recognition during the process
        if "face" in step.lower() or "analyzing" in step.lower():
            result = simulate_face_recognition()
            face_results.append(result)
        
        time.sleep(1.5)
    
    # Show final results
    final_result = face_results[-1] if face_results else simulate_face_recognition()
    
    with results_container.container():
        if final_result["faces_detected"] > 0:
            if final_result["person_identified"]:
                st.success("✅ Face recognition successful!")
                st.write(f"**Confidence:** {final_result['confidence']:.2%}")
                st.write(f"**Match Quality:** {final_result['match_quality']}")
                
                # Use employee name from login
                name = emp['name']
                gender = random.choice(["Male", "Female"])
                
                time.sleep(2)
                complete_attendance_process(emp, name, gender)
            else:
                st.warning("⚠️ Face detected but not recognized")
                st.session_state.unknown_face_detected = True
                time.sleep(2)
                st.rerun()
        else:
            st.error("❌ No face detected. Please ensure good lighting and try again.")
            if st.button("Try Again", key="retry_attendance"):
                st.session_state.analysis_in_progress = False
                st.rerun()

def complete_attendance_process(emp, name, gender):
    """Complete the attendance process"""
    st.session_state.basic_info_collected = True
    st.session_state.employee_info['detected_name'] = name
    st.session_state.employee_info['detected_gender'] = gender
    
    # Record attendance
    success = record_basic_attendance(
        emp['meeting_id'],
        emp['emp_id'],
        name,
        gender
    )
    
    if success:
        st.success("✅ Attendance recorded successfully!")
        st.info("🎥 Preparing secure meeting session...")
        
        # Show meeting preparation
        prep_steps = [
            "Initializing secure connection...",
            "Setting up video encryption...",
            "Configuring behavior monitoring...",
            "Starting meeting session..."
        ]
        
        for step in prep_steps:
            st.write(f"🔧 {step}")
            time.sleep(1)
        
        time.sleep(2)
        st.session_state.in_video_call = True
        st.session_state.meeting_start_time = time.time()
        st.rerun()
    else:
        st.error("Failed to record attendance. Please try again.")

def video_call_session():
    """Video call session with behavior monitoring"""
    
    # Show unknown face registration interface if needed
    if st.session_state.unknown_face_detected and st.session_state.capture_unknown_face:
        capture_unknown_face_interface()
        return
    
    emp = st.session_state.employee_info
    
    # Sidebar with meeting info
    with st.sidebar:
        st.title("🎥 Meeting Session")
        st.write(f"**Participant:** {emp.get('detected_name', 'Unknown')}")
        st.write(f"**Meeting ID:** {emp['meeting_id']}")
        st.write(f"**Employee ID:** {emp['emp_id']}")
        
        # Meeting duration
        if st.session_state.meeting_start_time:
            duration = int(time.time() - st.session_state.meeting_start_time)
            st.write(f"**Duration:** {duration // 60}:{duration % 60:02d}")
        
        st.markdown("---")
        
        # Behavior alerts
        if st.session_state.suspicious_moments:
            st.warning(f"🚨 {len(st.session_state.suspicious_moments)} behavior alerts")
        else:
            st.success("✅ No behavior alerts")
        
        if st.button("📞 End Meeting", type="primary", use_container_width=True):
            end_call_button(emp)
    
    st.title("🎥 Secure Meeting Session")
    
    # Main meeting interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Video feeds
        st.subheader("Video Fe
