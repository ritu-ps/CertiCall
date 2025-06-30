import cv2
import numpy as np
import os
import threading
import sounddevice as sd
import tempfile
import scipy.io.wavfile as wavfile
from datetime import datetime
from face_features import detect_faces, extract_emotion
from voice_features import extract_voice_features

# Initialize models and variables
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
age_net = gender_net = None
age_list = gender_list = []
face_dataset = face_labels = []
names = {}
lie_signs = 0
audio_lie_signs = 0
frame_count = 0
current_analysis = {
    "name": None,
    "gender": None,
    "lie_detected": False,
    "lie_timestamps": []
}

def load_models():
    global age_net, gender_net, age_list, gender_list, face_dataset, face_labels, names
    
    # Load age and gender models
    age_net = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')
    age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    gender_list = ['Male', 'Female']
    
    # Load face dataset
    dataset_path = './face_dataset/'
    face_data = []
    labels = []
    class_id = 0
    
    for fx in os.listdir(dataset_path):
        if fx.endswith('.npy'):
            names[class_id] = fx[:-4]
            data_item = np.load(os.path.join(dataset_path, fx))
            face_data.append(data_item)
            labels.extend([class_id] * data_item.shape[0])
            class_id += 1
    
    face_dataset = np.concatenate(face_data, axis=0)
    face_labels = np.array(labels).reshape(-1, 1)

def distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))

def knn(train, test, k=5):
    distances = []
    for i in range(train.shape[0]):
        dist = distance(train[i, :-1], test)
        distances.append((dist, train[i, -1]))
    
    distances = sorted(distances, key=lambda x: x[0])[:k]
    labels = [item[1] for item in distances]
    unique_labels, counts = np.unique(labels, return_counts=True)
    return unique_labels[np.argmax(counts)]

def analyze_voice():
    global audio_lie_signs
    try:
        duration = 5  # seconds
        fs = 44100
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        wavfile.write("temp.wav", fs, audio)
        features = extract_voice_features("temp.wav")
        return features["pitch"] > 220  # Returns boolean
    except Exception as e:
        print(f"Voice analysis error: {e}")
        return False

def process_basic_info_frame(frame):
    """Process frame to collect only name and gender"""
    try:
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return frame, None, None
        
        # Process first face only
        x, y, w, h = faces[0]
        face_roi = frame[y:y+h, x:x+w]
        
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Face recognition
        resized_face = cv2.resize(face_roi, (100, 100)).flatten()
        identity = knn(np.hstack((face_dataset, face_labels)), resized_face)
        name = names.get(int(identity), "Unknown")
        
        # Gender detection
        blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.426, 87.769, 114.896), swapRB=False)
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[np.argmax(gender_preds[0])]
        
        # Update current analysis with basic info only
        current_analysis["name"] = str(name)
        current_analysis["gender"] = str(gender)
        
        # Display info on frame
        cv2.putText(frame, f"Name: {name}", (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Gender: {gender}", (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame, name, gender
    
    except Exception as e:
        print(f"Basic info processing error: {e}")
        return frame, None, None

def process_call_frame(frame):
    """Process frame during video call for lie detection"""
    try:
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return frame, False, None
        
        # Process first face only
        x, y, w, h = faces[0]
        face_roi = frame[y:y+h, x:x+w]
        
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Emotion and lie detection
        emotion = extract_emotion(face_roi)
        lie_detected = emotion in ['fear', 'disgust', 'sad']
        lie_info = None
        
        if lie_detected:
            timestamp = datetime.now().strftime("%H:%M:%S")
            lie_info = f"emotion:{emotion}"
            current_analysis["lie_detected"] = True
            current_analysis["lie_timestamps"].append((timestamp, lie_info))
        
        # Voice analysis (every 150 frames)
        global frame_count
        frame_count += 1
        if frame_count % 150 == 0:
            if analyze_voice():
                timestamp = datetime.now().strftime("%H:%M:%S")
                lie_detected = True
                lie_info = "voice_stress"
                current_analysis["lie_detected"] = True
                current_analysis["lie_timestamps"].append((timestamp, lie_info))
        
        # Display lie detection info on frame
        if lie_detected:
            cv2.putText(frame, f"Alert: {lie_info}", (x, y-60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame, lie_detected, lie_info
    
    except Exception as e:
        print(f"Call frame processing error: {e}")
        return frame, False, None

def process_frame(frame):
    """Legacy function - kept for compatibility"""
    processed_frame, _, _ = process_basic_info_frame(frame)
    return processed_frame

def get_analysis_results():
    """Return the current analysis results"""
    return (
        current_analysis["name"],
        current_analysis["gender"],
        current_analysis["lie_detected"],
        current_analysis["lie_timestamps"]
    )

def reset_analysis():
    """Reset the analysis results"""
    global current_analysis, frame_count
    current_analysis = {
        "name": None,
        "gender": None,
        "lie_detected": False,
        "lie_timestamps": []
    }
    frame_count = 0

# Initialize models when module loads
load_models()
