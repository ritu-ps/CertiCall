import streamlit as st

# Page config must be the first Streamlit command
st.set_page_config(page_title="CertiCall", layout="wide")

# Now import other modules
import cv2
import time
from datetime import datetime
import database as db
import face_recog
import tempfile
import pyperclip
import sqlite3
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# Initialize database
db.init_db()

# WebRTC configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

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
def show_login_page():
    """Show login options for both host and employee"""
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
            
            st.success(f"Meeting created successfully!")
            
            # Display sharing options
            st.subheader("Share Meeting Access")
            st.markdown("Share this Meeting ID with participants:")
            
            # Create a box with meeting ID and copy button
            col1, col2 = st.columns([3,1])
            with col1:
                st.code(f"Meeting ID: {meeting_id}", language="text")
            with col2:
                if st.button("📋 Copy", key=f"copy_meeting_{meeting_id}"):
                    pyperclip.copy(str(meeting_id))
                    st.success("Copied to clipboard!")
    
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
                    
                    # Display credentials for sharing
                    st.subheader("Share Credentials")
                    st.markdown(f"""
                    **Share these credentials with {emp_name}:**
                    - **Meeting ID:** `{meeting_id}`
                    - **Employee ID:** `{emp_id}`
                    - **Password:** `{emp_password}`
                    """)
                    
                    # Create copy buttons
                    cols = st.columns(3)
                    with cols[0]:
                        if st.button(f"📋 Meeting ID", key=f"copy_mid_{emp_id}"):
                            pyperclip.copy(str(meeting_id))
                            st.toast("Meeting ID copied!")
                    with cols[1]:
                        if st.button(f"📋 Employee ID", key=f"copy_eid_{emp_id}"):
                            pyperclip.copy(emp_id)
                            st.toast("Employee ID copied!")
                    with cols[2]:
                        if st.button(f"📋 Password", key=f"copy_pwd_{emp_id}"):
                            pyperclip.copy(emp_password)
                            st.toast("Password copied!")
                else:
                    st.error("Employee ID already exists for this meeting")
            
            st.subheader("Current Employees")
            employees = db.get_employees_for_meeting(meeting_id)
            if employees:
                for emp_id, name in employees:
                    with st.expander(f"{name} (ID: {emp_id})"):
                        # Get employee details
                        conn = sqlite3.connect('meetings.db')
                        c = conn.cursor()
                        c.execute("SELECT password FROM employees WHERE meeting_id=? AND emp_id=?", 
                                 (meeting_id, emp_id))
                        password = c.fetchone()[0]
                        conn.close()
                        
                        st.markdown("**Credentials to share:**")
                        st.code(f"Meeting ID: {meeting_id}\nEmployee ID: {emp_id}\nPassword: {password}", 
                               language="text")
                        
                        if st.button("📋 Copy All", key=f"copy_all_{emp_id}"):
                            creds = f"Meeting ID: {meeting_id}\nEmployee ID: {emp_id}\nPassword: {password}"
                            pyperclip.copy(creds)
                            st.toast("All credentials copied!")
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
                    # Safely format the join time
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
    
    st.info("""
    **Instructions for Attendance:**
    1. Ensure good lighting and face the camera directly
    2. Remove any face coverings (masks, sunglasses)
    3. Speak clearly when prompted
    4. Remain still during the analysis
    """)
    
    if not st.session_state.analysis_in_progress:
        if st.button("Begin Attendance Check", key="begin_attendance_check"):
            st.session_state.analysis_in_progress = True
            st.rerun()
    else:
        perform_attendance_check()

def perform_attendance_check():
    """Perform the basic attendance check to collect name and gender"""
    emp = st.session_state.employee_info
    
    # Reset previous analysis
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
    analysis_duration = 10  # 10 seconds for basic info collection
    
    # Display countdown
    countdown_placeholder = st.empty()
    
    while not stop_button and (time.time() - start_time) < analysis_duration:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video frame")
            break
            
        # Flip the frame horizontally (mirror effect)
        frame = cv2.flip(frame, 1)
            
        # Process frame for basic info only
        processed_frame, name, gender = face_recog.process_basic_info_frame(frame)
        
        # Display the processed frame
        st_frame.image(processed_frame, channels="BGR", use_container_width=True)
        
        # Update countdown
        remaining_time = max(0, analysis_duration - (time.time() - start_time))
        countdown_placeholder.write(f"Time remaining: {int(remaining_time)} seconds")
        
        time.sleep(0.1)
    
    cap.release()
    st_frame.empty()
    countdown_placeholder.empty()
    
    if stop_button:
        st.warning("Attendance check cancelled")
        st.session_state.analysis_in_progress = False
        return
    
    if name and gender:
        st.session_state.basic_info_collected = True
        st.session_state.employee_info['detected_name'] = name
        st.session_state.employee_info['detected_gender'] = gender
        
        # Record basic attendance first
        db.record_basic_attendance(
            emp['meeting_id'],
            emp['emp_id'],
            name,
            gender
        )
        
        st.success("Basic information collected. Starting video call...")
        time.sleep(2)
        st.session_state.in_video_call = True
        st.rerun()
    else:
        st.error("Face recognition failed. Please ensure good lighting and try again.")
        st.session_state.analysis_in_progress = False

def video_call_session():
    emp = st.session_state.employee_info
    st.title("Video Call Session")
    st.write(f"**Name:** {emp.get('detected_name', 'Unknown')}")
    st.write(f"**Gender:** {emp.get('detected_gender', 'Unknown')}")

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
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)

            processed_img, lie_detected, lie_info = face_recog.process_call_frame(img)
            if lie_detected:
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.suspicious_moments.append((timestamp, lie_info))
            return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

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

    if st.button("End Call"):
        if st.session_state.suspicious_moments:
            db.update_suspicious_moments(
                emp['meeting_id'],
                emp['emp_id'],
                str(st.session_state.suspicious_moments)
            )
        st.success("Call ended. Thank you for your participation.")
        time.sleep(2)
        st.session_state.logged_in = False
        st.session_state.user_type = None
        st.session_state.employee_info = None
        st.session_state.analysis_in_progress = False
        st.session_state.in_video_call = False
        st.session_state.basic_info_collected = False
        st.session_state.suspicious_moments = []
        st.session_state.camera_on = True
        st.session_state.mic_on = True
        st.rerun()

    if webrtc_ctx and not webrtc_ctx.state.playing:
        st.warning("Connection lost. Please wait...")
        time.sleep(1)
        st.rerun()


def main():
    st.title("CertiCall")

    if not st.session_state.logged_in:
        show_login_page()
    else:
        if st.session_state.user_type == 'host':
            host_dashboard()
        else:
            employee_interface()


if __name__ == "__main__":
    main()