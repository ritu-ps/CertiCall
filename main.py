import streamlit as st
import tempfile
import time
from datetime import datetime
import database as db
import sqlite3
import numpy as np
from PIL import Image
import os

# Page config must be the first Streamlit command
st.set_page_config(page_title="CertiCall", layout="wide")

# Import handling for Streamlit Cloud limitations
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import face_recog
    FACE_RECOG_AVAILABLE = True
except ImportError:
    FACE_RECOG_AVAILABLE = False

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

# Environment detection
IS_STREAMLIT_CLOUD = not CV2_AVAILABLE  # Simple heuristic

# Initialize database
db.init_db()

# WebRTC configuration (if available)
if WEBRTC_AVAILABLE:
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
else:
    RTC_CONFIGURATION = None

# Session state
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
if 'camera_on' not in st.session_state:
    st.session_state.camera_on = True
if 'mic_on' not in st.session_state:
    st.session_state.mic_on = True
if 'video_call_key' not in st.session_state:
    st.session_state.video_call_key = "video-call"
if 'unknown_face_detected' not in st.session_state:
    st.session_state.unknown_face_detected = False
if 'capture_unknown_face' not in st.session_state:
    st.session_state.capture_unknown_face = False
if 'unknown_face_image' not in st.session_state:
    st.session_state.unknown_face_image = None
if 'unknown_face_name' not in st.session_state:
    st.session_state.unknown_face_name = ""

def save_unknown_face_to_dataset(face_image, person_name, meeting_id):
    """Save unknown face to the face_dataset directory"""
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
        filename = f"{person_name}_{meeting_id}_{timestamp}.jpg"
        filepath = os.path.join(person_dir, filename)
        
        # Save the face image
        if isinstance(face_image, np.ndarray):
            cv2.imwrite(filepath, face_image)
        else:
            face_image.save(filepath)
        
        # Also save to a general unknown faces log
        log_file = os.path.join(dataset_dir, "unknown_faces_log.csv")
        with open(log_file, 'a') as f:
            f.write(f"{timestamp},{person_name},{meeting_id},{filepath}\n")
        
        return True, filepath
    except Exception as e:
        return False, str(e)

def capture_unknown_face_interface():
    """Interface for capturing and registering unknown faces"""
    st.warning("🔍 Unknown Face Detected!")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.session_state.unknown_face_image is not None:
            st.image(st.session_state.unknown_face_image, caption="Detected Unknown Face", use_container_width=True)
    
    with col2:
        st.subheader("Register New Person")
        person_name = st.text_input("Enter person's name:", 
                                   value=st.session_state.unknown_face_name,
                                   key="unknown_face_name_input")
        
        if st.button("Save Face to Dataset", type="primary"):
            if person_name.strip():
                emp = st.session_state.employee_info
                success, result = save_unknown_face_to_dataset(
                    st.session_state.unknown_face_image, 
                    person_name.strip(),
                    emp['meeting_id']
                )
                
                if success:
                    st.success(f"✅ Face saved for {person_name}!")
                    st.info(f"Image saved to: {result}")
                    
                    # Update the current session with the new name
                    st.session_state.employee_info['detected_name'] = person_name.strip()
                    st.session_state.unknown_face_detected = False
                    st.session_state.capture_unknown_face = False
                    st.session_state.unknown_face_image = None
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
        
        if st.button("Skip Registration"):
            st.session_state.unknown_face_detected = False
            st.session_state.capture_unknown_face = False
            st.session_state.unknown_face_image = None
            st.session_state.unknown_face_name = ""
            st.rerun()

def show_environment_warning():
    """Show warnings about Streamlit Cloud limitations"""
    if IS_STREAMLIT_CLOUD:
        st.warning("""
        ⚠️ **Streamlit Cloud Limitations**:
        - Camera access not available
        - Real-time video processing disabled
        - Clipboard copying requires manual selection
        - Some advanced features simulated
        """)

def show_login_page():
    """Show login options for both host and employee"""
    show_environment_warning()
    
    tab1, tab2 = st.tabs(["Host Portal", "Employee Portal"])
    
    with tab1:
        st.header("Host Authentication")
        login_tab, register_tab = st.tabs(["Login", "Register"])
        
        with login_tab:
            st.subheader("Host Login")
            email = st.text_input("Email", key="host_login_email")
            password = st.text_input("Password", type="password", key="host_login_password")
            
            if st.button("Login as Host", key="host_login_button"):
                host_info = db.verify_host(email, password)
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
            
            if st.button("Register as Host", key="host_register_button"):
                if password != confirm_password:
                    st.error("Passwords do not match")
                elif not all([name, company, email, password]):
                    st.error("Please fill all fields")
                else:
                    if db.add_host(name, email, password, company):
                        st.success("Registration successful! Please login.")
                        # Auto-login after registration
                        st.session_state.logged_in = True
                        st.session_state.user_type = 'host'
                        st.session_state.host_info = {
                            "id": db.verify_host(email, password)[0],
                            "name": name,
                            "company": company
                        }
                        st.rerun()
                    else:
                        st.error("Email already registered")

    with tab2:
        st.header("Employee Login")
        meeting_id = st.text_input("Meeting ID", key="emp_meeting_id")
        emp_id = st.text_input("Employee ID", key="emp_id")
        password = st.text_input("Password", type="password", key="emp_password")
        
        if st.button("Join Meeting", key="emp_login_button"):
            if not all([meeting_id, emp_id, password]):
                st.error("Please fill all fields")
            else:
                employee_info = db.verify_employee(meeting_id, emp_id, password)
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
    st.sidebar.title(f"Host Portal")
    st.sidebar.subheader(f"{host['company']}")
    st.sidebar.write(f"Welcome, {host['name']}")
    
    if st.sidebar.button("Logout", key="host_logout_button"):
        st.session_state.logged_in = False
        st.session_state.user_type = None
        st.session_state.host_info = None
        st.rerun()
    
    tab1, tab2, tab3 = st.tabs(["Create Meeting", "Manage Employees", "View Attendance"])
    
    with tab1:
        st.header("Create New Meeting")
        title = st.text_input("Meeting Title", key="meeting_title")
        description = st.text_area("Description", key="meeting_description")
        start_time = st.date_input("Date", key="meeting_date")
        start_hour = st.time_input("Start Time", key="meeting_start_time")
        end_hour = st.time_input("End Time (optional)", value=None, key="meeting_end_time")
        
        if st.button("Create Meeting", key="create_meeting_button"):
            start_datetime = datetime.combine(start_time, start_hour)
            end_datetime = datetime.combine(start_time, end_hour) if end_hour else None
            meeting_id = db.create_meeting(host['id'], title, description, start_datetime, end_datetime)
            st.session_state.current_meeting = meeting_id
            
            st.success(f"Meeting created successfully! Meeting ID: {meeting_id}")
            
            # Display sharing options
            st.subheader("Share Meeting Access")
            st.markdown("Share this Meeting ID with participants:")
            st.code(f"Meeting ID: {meeting_id}", language="text")
    
    with tab2:
        st.header("Manage Employees")
        meetings = db.get_meetings_for_host(host['id'])
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
            
            if st.button("Add Employee", key="add_employee_button"):
                if db.add_employee(meeting_id, emp_name, emp_id, emp_password):
                    st.success(f"Employee {emp_name} added successfully!")
                else:
                    st.error("Employee ID already exists for this meeting")
            
            st.subheader("Current Employees")
            employees = db.get_employees_for_meeting(meeting_id)
            if employees:
                for emp_id, name in employees:
                    with st.expander(f"{name} (ID: {emp_id})"):
                        st.code(f"Meeting ID: {meeting_id}\nEmployee ID: {emp_id}", language="text")
            else:
                st.info("No employees added yet")

    with tab3:
        st.header("View Attendance")
        meetings = db.get_meetings_for_host(host['id'])
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
            
            attendance = db.get_attendance_for_meeting(meeting_id)
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
                            st.error("**Lie Detection Alert!**")
                            if lie_timestamps:
                                try:
                                    timestamps = eval(lie_timestamps) if isinstance(lie_timestamps, str) else lie_timestamps
                                    st.write("**Suspicious moments:**")
                                    for ts in timestamps:
                                        st.write(f"- {ts[0]}: {ts[1]}")
                                except:
                                    st.warning("Could not parse lie timestamps")
                        else:
                            st.success("No suspicious behavior detected")
            else:
                st.info("No attendance records yet")

def employee_interface():
    """Employee interface after joining meeting"""
    if st.session_state.in_video_call:
        video_call_session()
        return
    
    emp = st.session_state.employee_info
    st.title(f"Meeting Attendance Portal")
    st.subheader(f"Welcome, {emp['name']}")
    
    show_environment_warning()
    
    st.info("""
    **Instructions for Attendance:**
    1. Ensure good lighting and face the camera directly
    2. Remove any face coverings (masks, sunglasses)
    3. Speak clearly when prompted
    4. Remain still during the analysis
    """)
    
    if IS_STREAMLIT_CLOUD:
        st.warning("""
        🔄 **Streamlit Cloud Mode**: 
        Camera-based attendance is simulated. 
        You'll proceed directly to the meeting session.
        """)
    
    if not st.session_state.analysis_in_progress:
        if st.button("Begin Attendance Check", key="begin_attendance_check"):
            st.session_state.analysis_in_progress = True
            st.rerun()
    else:
        perform_attendance_check()

def perform_attendance_check():
    """Perform the basic attendance check - simulated on Streamlit Cloud"""
    emp = st.session_state.employee_info
    
    if IS_STREAMLIT_CLOUD or not CV2_AVAILABLE:
        # Streamlit Cloud - simulate the process
        perform_attendance_check_simulated()
        return
    
    # Local environment with camera access
    try:
        # Reset previous analysis
        if FACE_RECOG_AVAILABLE:
            face_recog.reset_analysis()
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not access camera. Please check permissions.")
            st.session_state.analysis_in_progress = False
            return
        
        st_frame = st.empty()
        stop_button = st.button("Cancel Analysis", key="cancel_analysis_button")
        
        start_time = time.time()
        analysis_duration = 10
        
        progress_bar = st.progress(0)
        name, gender = emp['name'], "Unknown"
        
        while not stop_button and (time.time() - start_time) < analysis_duration:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video frame")
                break
                
            frame = cv2.flip(frame, 1)
                
            if FACE_RECOG_AVAILABLE:
                processed_frame, detected_name, detected_gender = face_recog.process_basic_info_frame(frame)
                if detected_name and detected_name != "Unknown":
                    name, gender = detected_name, detected_gender
            else:
                processed_frame = frame
            
            st_frame.image(processed_frame, channels="BGR", use_container_width=True)
            
            elapsed_time = time.time() - start_time
            progress = min(elapsed_time / analysis_duration, 1.0)
            progress_bar.progress(progress)
            
            time.sleep(0.1)
        
        cap.release()
        st_frame.empty()
        progress_bar.empty()
        
        if stop_button:
            st.warning("Attendance check cancelled")
            st.session_state.analysis_in_progress = False
            return
        
    except Exception as e:
        st.error(f"Camera error: {str(e)}")
        st.session_state.analysis_in_progress = False
        return
    
    complete_attendance_process(emp, name, gender)

def perform_attendance_check_simulated():
    """Simulated attendance check for Streamlit Cloud"""
    emp = st.session_state.employee_info
    
    st.info("🔍 **Simulating Face Recognition...**")
    
    progress_bar = st.progress(0)
    for i in range(5):
        progress_bar.progress((i + 1) * 20)
        time.sleep(0.5)
    
    name = emp['name']
    gender = "Unknown"
    
    complete_attendance_process(emp, name, gender)

def complete_attendance_process(emp, name, gender):
    """Complete the attendance process"""
    st.session_state.basic_info_collected = True
    st.session_state.employee_info['detected_name'] = name
    st.session_state.employee_info['detected_gender'] = gender
    
    db.record_basic_attendance(
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
    
    if IS_STREAMLIT_CLOUD or not WEBRTC_AVAILABLE:
        video_call_simulation()
        return
    
    # Local environment with WebRTC
    st.title("Video Call Session")
    
    # Show current detected name with option to change if unknown
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write(f"**Current Participant:** {emp.get('detected_name', 'Unknown')}")
        st.write(f"**Gender:** {emp.get('detected_gender', 'Unknown')}")
    
    with col2:
        if st.button("🚨 Register Unknown Face", type="secondary"):
            st.session_state.capture_unknown_face = True
            st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Toggle Camera"):
            st.session_state.camera_on = not st.session_state.camera_on
            st.session_state.video_call_key = str(time.time())
            st.rerun()
    with col2:
        if st.button("Toggle Microphone"):
            st.session_state.mic_on = not st.session_state.mic_on
            st.session_state.video_call_key = str(time.time())
            st.rerun()

    class VideoProcessor:
        def __init__(self):
            self.unknown_face_counter = 0
            self.last_unknown_detection = 0
            
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)

            if FACE_RECOG_AVAILABLE and CV2_AVAILABLE:
                processed_img, lie_detected, lie_info, unknown_face_detected = face_recog.process_call_frame(img)
                
                # Handle unknown face detection
                if unknown_face_detected and (time.time() - self.last_unknown_detection) > 10:
                    self.unknown_face_counter += 1
                    self.last_unknown_detection = time.time()
                    
                    # Only trigger after multiple detections to avoid false positives
                    if self.unknown_face_counter >= 3 and not st.session_state.unknown_face_detected:
                        st.session_state.unknown_face_detected = True
                        st.session_state.unknown_face_image = processed_img
                        # Store the frame for registration
                        st.session_state.unknown_face_image = img.copy()
                
                if lie_detected:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    st.session_state.suspicious_moments.append((timestamp, lie_info))
                
                return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
            else:
                return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key=st.session_state.video_call_key,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": st.session_state.camera_on,
            "audio": st.session_state.mic_on
        },
        async_processing=True,
    )

    # Show unknown face alert if detected
    if st.session_state.unknown_face_detected and not st.session_state.capture_unknown_face:
        st.warning("🔍 Unknown face detected! Click 'Register Unknown Face' to add to dataset.")
    
    end_call_button(emp)

    if webrtc_ctx and not webrtc_ctx.state.playing:
        st.warning("Connection lost. Please wait...")
        time.sleep(1)
        st.rerun()

def video_call_simulation():
    """Simulated video call for Streamlit Cloud"""
    emp = st.session_state.employee_info
    
    st.title("Meeting Session - Simulation Mode")
    st.warning("🎥 **Video Call Simulation** - Real video streaming not available on Streamlit Cloud")
    
    # Unknown face simulation
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write(f"**Participant:** {emp.get('detected_name', 'Unknown')}")
        st.write(f"**Meeting ID:** {emp['meeting_id']}")
    
    with col2:
        if st.button("🚨 Simulate Unknown Face", type="secondary"):
            st.session_state.unknown_face_detected = True
            st.session_state.capture_unknown_face = True
            # Create a dummy image for simulation
            dummy_image = np.ones((200, 200, 3), dtype=np.uint8) * 128
            st.session_state.unknown_face_image = dummy_image
            st.rerun()
    
    # Show unknown face registration if triggered
    if st.session_state.unknown_face_detected and st.session_state.capture_unknown_face:
        capture_unknown_face_interface()
        return
    
    # Simulate meeting duration
    if 'call_start_time' not in st.session_state:
        st.session_state.call_start_time = time.time()
        st.session_state.last_lie_check = time.time()
    
    call_duration = int(time.time() - st.session_state.call_start_time)
    st.write(f"**Meeting Duration:** {call_duration} seconds")
    
    # Simulate occasional lie detection
    current_time = time.time()
    if current_time - st.session_state.last_lie_check > 15:
        if np.random.random() < 0.3:
            timestamp = datetime.now().strftime("%H:%M:%S")
            behaviors = ["Unusual eye movement", "Voice stress detected", "Inconsistent head movement"]
            behavior = np.random.choice(behaviors)
            st.session_state.suspicious_moments.append((timestamp, behavior))
            st.warning(f"⚠️ Suspicious behavior detected at {timestamp}: {behavior}")
        st.session_state.last_lie_check = current_time
    
    # Display current suspicious moments
    if st.session_state.suspicious_moments:
        with st.expander("📊 Behavior Analysis"):
            st.write("**Suspicious moments detected:**")
            for timestamp, behavior in st.session_state.suspicious_moments[-5:]:
                st.write(f"- {timestamp}: {behavior}")
    
    end_call_button(emp)

def end_call_button(emp):
    """Common end call button for both real and simulated calls"""
    if st.button("End Meeting", type="primary", use_container_width=True):
        if st.session_state.suspicious_moments:
            db.update_suspicious_moments(
                emp['meeting_id'],
                emp['emp_id'],
                str(st.session_state.suspicious_moments)
            )
        
        st.success("✅ Meeting completed successfully!")
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
    st.session_state.camera_on = True
    st.session_state.mic_on = True
    st.session_state.unknown_face_detected = False
    st.session_state.capture_unknown_face = False
    st.session_state.unknown_face_image = None
    st.session_state.unknown_face_name = ""
    if 'call_start_time' in st.session_state:
        del st.session_state.call_start_time
    if 'last_lie_check' in st.session_state:
        del st.session_state.last_lie_check

def main():
    st.title("🎯 CertiCall - Secure Meeting Authentication")
    
    if IS_STREAMLIT_CLOUD:
        st.sidebar.info("🌐 **Streamlit Cloud Mode**")
    else:
        st.sidebar.success("💻 **Local Environment** - Full features available")

    if not st.session_state.logged_in:
        show_login_page()
    else:
        if st.session_state.user_type == 'host':
            host_dashboard()
        else:
            employee_interface()

if __name__ == "__main__":
    main()
