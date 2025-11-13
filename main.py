import streamlit as st
import time
from datetime import datetime
import json
import os
import hashlib
import sqlite3

# Page config must be the first Streamlit command
st.set_page_config(page_title="CertiCall", layout="wide")

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
                      company TEXT NOT NULL)''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS meetings
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      host_id INTEGER,
                      title TEXT NOT NULL,
                      description TEXT,
                      start_time TIMESTAMP,
                      end_time TIMESTAMP)''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS employees
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      meeting_id INTEGER,
                      name TEXT NOT NULL,
                      emp_id TEXT NOT NULL,
                      password TEXT NOT NULL,
                      UNIQUE(meeting_id, emp_id))''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS attendance
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      meeting_id INTEGER,
                      emp_id TEXT NOT NULL,
                      name TEXT NOT NULL,
                      gender TEXT,
                      join_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      lie_detected BOOLEAN DEFAULT FALSE,
                      lie_timestamps TEXT)''')
        
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
        st.error(f"Error adding host: {e}")
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
        st.error(f"Error verifying host: {e}")
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
        st.error(f"Error creating meeting: {e}")
        return None

def get_meetings_for_host(host_id):
    """Get all meetings for a host"""
    try:
        conn = sqlite3.connect('meetings.db', check_same_thread=False)
        c = conn.cursor()
        c.execute("SELECT id, title FROM meetings WHERE host_id=?", (host_id,))
        results = c.fetchall()
        conn.close()
        return results
    except Exception as e:
        st.error(f"Error getting meetings: {e}")
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
        st.error(f"Error adding employee: {e}")
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
        st.error(f"Error getting employees: {e}")
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
        st.error(f"Error verifying employee: {e}")
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
        st.error(f"Error recording attendance: {e}")
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
        st.error(f"Error updating suspicious moments: {e}")
        return False

def get_attendance_for_meeting(meeting_id):
    """Get attendance records for a meeting"""
    try:
        conn = sqlite3.connect('meetings.db', check_same_thread=False)
        c = conn.cursor()
        c.execute('''SELECT emp_id, name, gender, join_time, lie_detected, lie_timestamps 
                     FROM attendance WHERE meeting_id=? ORDER BY join_time DESC''',
                 (meeting_id,))
        results = c.fetchall()
        conn.close()
        return results
    except Exception as e:
        st.error(f"Error getting attendance: {e}")
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
        
        # Save metadata
        metadata = {
            "name": person_name,
            "meeting_id": meeting_id,
            "timestamp": timestamp,
            "registered_at": datetime.now().isoformat(),
            "type": "simulated_registration"
        }
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
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

def capture_unknown_face_interface():
    """Interface for capturing and registering unknown faces"""
    st.warning("🔍 Unknown Face Detected!")
    
    st.subheader("Register New Person")
    person_name = st.text_input("Enter person's name:", 
                               value=st.session_state.unknown_face_name,
                               key="unknown_face_name_input")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Face to Dataset", type="primary", use_container_width=True):
            if person_name.strip():
                emp = st.session_state.employee_info
                success, result = save_unknown_face_to_dataset(
                    person_name.strip(),
                    emp['meeting_id']
                )
                
                if success:
                    st.success(f"✅ Face registered for {person_name}!")
                    st.info(f"Metadata saved to: {result}")
                    
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
    
    with col2:
        if st.button("⏭️ Skip Registration", use_container_width=True):
            st.session_state.unknown_face_detected = False
            st.session_state.capture_unknown_face = False
            st.session_state.unknown_face_name = ""
            st.rerun()

def show_login_page():
    """Show login options for both host and employee"""
    st.info("🌐 **Cloud Mode**: All features are simulated for demonstration")
    
    tab1, tab2 = st.tabs(["🏢 Host Portal", "👤 Employee Portal"])
    
    with tab1:
        st.header("Host Authentication")
        login_tab, register_tab = st.tabs(["🔐 Login", "📝 Register"])
        
        with login_tab:
            st.subheader("Host Login")
            email = st.text_input("Email", key="host_login_email")
            password = st.text_input("Password", type="password", key="host_login_password")
            
            if st.button("Login as Host", key="host_login_button", type="primary"):
                if not email or not password:
                    st.error("Please fill all fields")
                else:
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
            
            if st.button("Register as Host", key="host_register_button", type="primary"):
                if password != confirm_password:
                    st.error("Passwords do not match")
                elif not all([name, company, email, password]):
                    st.error("Please fill all fields")
                else:
                    if add_host(name, email, password, company):
                        st.success("Registration successful! Please login.")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Email already registered")

    with tab2:
        st.header("Employee Login")
        meeting_id = st.text_input("Meeting ID", key="emp_meeting_id")
        emp_id = st.text_input("Employee ID", key="emp_id")
        password = st.text_input("Password", type="password", key="emp_password")
        
        if st.button("Join Meeting", key="emp_login_button", type="primary"):
            if not all([meeting_id, emp_id, password]):
                st.error("Please fill all fields")
            else:
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
    st.sidebar.title(f"🏢 Host Portal")
    st.sidebar.subheader(f"{host['company']}")
    st.sidebar.write(f"Welcome, {host['name']}")
    
    if st.sidebar.button("🚪 Logout", key="host_logout_button"):
        st.session_state.logged_in = False
        st.session_state.user_type = None
        st.session_state.host_info = None
        st.rerun()
    
    tab1, tab2, tab3 = st.tabs(["📅 Create Meeting", "👥 Manage Employees", "📊 View Attendance"])
    
    with tab1:
        st.header("Create New Meeting")
        col1, col2 = st.columns(2)
        with col1:
            title = st.text_input("Meeting Title", key="meeting_title")
            start_time = st.date_input("Date", key="meeting_date")
        with col2:
            description = st.text_area("Description", key="meeting_description")
            start_hour = st.time_input("Start Time", key="meeting_start_time")
        
        end_hour = st.time_input("End Time (optional)", value=None, key="meeting_end_time")
        
        if st.button("Create Meeting", key="create_meeting_button", type="primary"):
            if not title:
                st.error("Please enter a meeting title")
            else:
                start_datetime = datetime.combine(start_time, start_hour)
                end_datetime = datetime.combine(start_time, end_hour) if end_hour else None
                meeting_id = create_meeting(host['id'], title, description, start_datetime, end_datetime)
                st.session_state.current_meeting = meeting_id
                
                st.success(f"✅ Meeting created successfully! Meeting ID: {meeting_id}")
                
                # Display sharing options
                st.subheader("Share Meeting Access")
                st.markdown("Share this Meeting ID with participants:")
                st.code(f"Meeting ID: {meeting_id}", language="text")
    
    with tab2:
        st.header("Manage Employees")
        meetings = get_meetings_for_host(host['id'])
        if not meetings:
            st.warning("No meetings found. Please create a meeting first.")
        else:
            meeting_options = {f"{m[1]} (ID: {m[0]})": m[0] for m in meetings}
            selected_meeting = st.selectbox(
                "Select Meeting", 
                options=list(meeting_options.keys()),
                key="manage_employees_select_meeting"
            )
            meeting_id = meeting_options[selected_meeting]
            
            st.subheader("Add New Employee")
            col1, col2, col3 = st.columns(3)
            with col1:
                emp_name = st.text_input("Employee Name", key="add_emp_name")
            with col2:
                emp_id = st.text_input("Employee ID", key="add_emp_id")
            with col3:
                emp_password = st.text_input("Password", type="password", key="add_emp_password")
            
            if st.button("Add Employee", key="add_employee_button", type="primary"):
                if not all([emp_name, emp_id, emp_password]):
                    st.error("Please fill all fields")
                else:
                    if add_employee(meeting_id, emp_name, emp_id, emp_password):
                        st.success(f"✅ Employee {emp_name} added successfully!")
                    else:
                        st.error("Employee ID already exists for this meeting")
            
            st.subheader("Current Employees")
            employees = get_employees_for_meeting(meeting_id)
            if employees:
                for emp_id, name in employees:
                    with st.expander(f"{name} (ID: {emp_id})"):
                        st.code(f"Meeting ID: {meeting_id}\nEmployee ID: {emp_id}", language="text")
            else:
                st.info("No employees added yet")

    with tab3:
        st.header("View Attendance")
        meetings = get_meetings_for_host(host['id'])
        if not meetings:
            st.warning("No meetings found.")
        else:
            meeting_options = {f"{m[1]} (ID: {m[0]})": m[0] for m in meetings}
            selected_meeting = st.selectbox(
                "Select Meeting", 
                options=list(meeting_options.keys()),
                key="view_attendance_select_meeting"
            )
            meeting_id = meeting_options[selected_meeting]
            
            attendance = get_attendance_for_meeting(meeting_id)
            if attendance:
                st.subheader("Attendance Records")
                for emp_id, name, gender, join_time, lie_detected, lie_timestamps in attendance:
                    try:
                        if isinstance(join_time, str):
                            join_time_str = join_time
                        else:
                            join_time_str = join_time.strftime('%Y-%m-%d %H:%M:%S')
                    except Exception as e:
                        join_time_str = str(join_time)
                    
                    with st.expander(f"{name} ({gender}) - {join_time_str}"):
                        st.write(f"**Employee ID:** {emp_id}")
                        st.write(f"**Join Time:** {join_time_str}")
                        
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
                st.info("No attendance records yet")

def employee_interface():
    """Employee interface after joining meeting"""
    if st.session_state.in_video_call:
        video_call_session()
        return
    
    emp = st.session_state.employee_info
    st.title(f"🎯 Meeting Attendance Portal")
    st.subheader(f"Welcome, {emp['name']}")
    
    st.info("""
    **Instructions for Attendance:**
    1. Click 'Begin Attendance Check' to start
    2. The system will simulate face recognition
    3. You'll then proceed to the meeting session
    """)
    
    if not st.session_state.analysis_in_progress:
        if st.button("Begin Attendance Check", key="begin_attendance_check", type="primary"):
            st.session_state.analysis_in_progress = True
            st.rerun()
    else:
        perform_attendance_check()

def perform_attendance_check():
    """Simulated attendance check"""
    emp = st.session_state.employee_info
    
    st.info("🔍 **Simulating Face Recognition...**")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    steps = [
        "Initializing camera...",
        "Detecting face...", 
        "Analyzing facial features...",
        "Identifying person...",
        "Finalizing recognition..."
    ]
    
    for i, step in enumerate(steps):
        progress_bar.progress((i + 1) * 20)
        status_text.text(f"{step} {((i + 1) * 20)}%")
        time.sleep(1)
    
    # Use employee name from login
    name = emp['name']
    gender = "Unknown"
    
    status_text.text("✅ Face recognition complete!")
    time.sleep(1)
    
    complete_attendance_process(emp, name, gender)

def complete_attendance_process(emp, name, gender):
    """Complete the attendance process"""
    st.session_state.basic_info_collected = True
    st.session_state.employee_info['detected_name'] = name
    st.session_state.employee_info['detected_gender'] = gender
    
    record_basic_attendance(
        emp['meeting_id'],
        emp['emp_id'],
        name,
        gender
    )
    
    st.success("✅ Basic information collected successfully!")
    st.info("🎥 Starting meeting session...")
    time.sleep(2)
    st.session_state.in_video_call = True
    st.rerun()

def video_call_session():
    """Video call session with unknown face detection"""
    
    # Show unknown face registration interface if needed
    if st.session_state.unknown_face_detected and st.session_state.capture_unknown_face:
        capture_unknown_face_interface()
        return
    
    emp = st.session_state.employee_info
    
    st.title("🎥 Meeting Session")
    
    # Show current detected name with option to change if unknown
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write(f"**Participant:** {emp.get('detected_name', 'Unknown')}")
        st.write(f"**Meeting ID:** {emp['meeting_id']}")
        st.write(f"**Employee ID:** {emp['emp_id']}")
    
    with col2:
        if st.button("🚨 Register Unknown Face", type="secondary"):
            st.session_state.unknown_face_detected = True
            st.session_state.capture_unknown_face = True
            st.rerun()
    
    # Simulate meeting duration
    if 'call_start_time' not in st.session_state:
        st.session_state.call_start_time = time.time()
        st.session_state.last_behavior_check = time.time()
    
    call_duration = int(time.time() - st.session_state.call_start_time)
    st.write(f"**Meeting Duration:** {call_duration} seconds")
    
    # Meeting simulation
    st.subheader("Meeting Simulation")
    
    # Simulate video feed placeholder
    col1, col2 = st.columns(2)
    with col1:
        st.info("🎥 **Your Video Feed**")
        st.markdown("""
        <div style='background: linear-gradient(45deg, #4CAF50, #45a049); padding: 20px; border-radius: 10px; text-align: center; color: white;'>
            <h3>📹 Your Camera Feed</h3>
            <p>Simulated video stream</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.info("👥 **Other Participants**")
        st.markdown("""
        <div style='background: linear-gradient(45deg, #2196F3, #1976D2); padding: 20px; border-radius: 10px; text-align: center; color: white;'>
            <h3>👤 Participant Video</h3>
            <p>Simulated participant stream</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Meeting controls
    st.subheader("Meeting Controls")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🎤 Mute/Unmute", use_container_width=True):
            st.toast("Microphone toggled")
    with col2:
        if st.button("📹 Start/Stop Video", use_container_width=True):
            st.toast("Video toggled")
    with col3:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    with col4:
        if st.button("📋 Share Screen", use_container_width=True):
            st.toast("Screen sharing activated")
    
    # Simulate occasional behavior detection
    current_time = time.time()
    if current_time - st.session_state.last_behavior_check > 10:
        if len(st.session_state.suspicious_moments) < 3:
            timestamp = datetime.now().strftime("%H:%M:%S")
            behaviors = [
                "Unusual eye movement pattern detected",
                "Voice stress analysis indicates nervousness", 
                "Inconsistent head movement detected"
            ]
            behavior = behaviors[len(st.session_state.suspicious_moments) % len(behaviors)]
            st.session_state.suspicious_moments.append((timestamp, behavior))
            st.warning(f"⚠️ Anomaly detected at {timestamp}")
        st.session_state.last_behavior_check = current_time
    
    # Display current suspicious moments
    if st.session_state.suspicious_moments:
        with st.expander("📊 Behavior Analysis Report"):
            st.write("**Anomalies detected during meeting:**")
            for timestamp, behavior in st.session_state.suspicious_moments:
                st.write(f"- **{timestamp}**: {behavior}")
    
    # Unknown face detection simulation
    if not st.session_state.unknown_face_detected and len(st.session_state.suspicious_moments) > 1:
        st.session_state.unknown_face_detected = True
        st.warning("🔍 Unknown face pattern detected! Consider registering this person.")
    
    end_call_button(emp)

def end_call_button(emp):
    """Common end call button"""
    if st.button("📞 End Meeting", type="primary", use_container_width=True):
        if st.session_state.suspicious_moments:
            update_suspicious_moments(
                emp['meeting_id'],
                emp['emp_id'],
                json.dumps(st.session_state.suspicious_moments)
            )
        
        st.success("✅ Meeting completed successfully!")
        st.balloons()
        time.sleep(2)
        reset_employee_session()
        st.rerun()

def reset_employee_session():
    """Reset all employee session variables"""
    st.session_state.logged_in = False
    st.session_state.user_type = None
    st.session_state.employee_info = None
    st.session_state.analysis_in_progress = False
    st.session_state.in_video_call = False
    st.session_state.basic_info_collected = False
    st.session_state.suspicious_moments = []
    st.session_state.unknown_face_detected = False
    st.session_state.capture_unknown_face = False
    st.session_state.unknown_face_name = ""
    if 'call_start_time' in st.session_state:
        del st.session_state.call_start_time
    if 'last_behavior_check' in st.session_state:
        del st.session_state.last_behavior_check

def main():
    st.title("🎯 CertiCall - Secure Meeting Authentication")
    st.sidebar.info("🌐 **Cloud Deployment** - All features simulated")
    
    # Initialize database on startup
    init_db()

    if not st.session_state.logged_in:
        show_login_page()
    else:
        if st.session_state.user_type == 'host':
            host_dashboard()
        else:
            employee_interface()

if __name__ == "__main__":
    main()
