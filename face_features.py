import cv2
from deepface import DeepFace

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces

def extract_emotion(face_img):
    try:
        # Analyze emotion using DeepFace
        analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
        emotion = analysis[0]['dominant_emotion']
        return emotion
    except Exception as e:
        print(f"[ERROR] Emotion detection failed: {e}")
        return "unknown"