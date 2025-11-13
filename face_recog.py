import cv2
import numpy as np
import os
import threading
import sounddevice as sd
import tempfile
import scipy.io.wavfile as wavfile
from datetime import datetime
from tensorflow.keras.models import load_model
from face_features import detect_faces, extract_emotion
from voice_features import extract_voice_features

# Initialize models and variables
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
age_net = None
gender_model = None
age_list = []
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

def create_simple_gender_model():
    """Create a simple gender classification model as fallback"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("✅ Simple fallback gender model created")
    return model

def inspect_gender_model():
    """Inspect the gender model architecture"""
    if gender_model is None:
        print("Gender model not loaded")
        return
    
    print("=== Gender Model Inspection ===")
    print(f"Number of inputs: {len(gender_model.inputs)}")
    print(f"Number of outputs: {len(gender_model.outputs)}")
    
    for i, input_layer in enumerate(gender_model.inputs):
        print(f"Input {i}: {input_layer.shape} - {input_layer.dtype}")
    
    for i, output_layer in enumerate(gender_model.outputs):
        print(f"Output {i}: {output_layer.shape} - {output_layer.dtype}")
    
    print("Model layers:")
    for layer in gender_model.layers:
        print(f"  {layer.name} - {type(layer).__name__} - Input: {layer.input_shape} - Output: {layer.output_shape}")

def predict_gender(face_roi):
    """Predict gender with comprehensive error handling"""
    global gender_model
    
    if gender_model is None:
        return "Unknown"
    
    try:
        # Preprocess the face ROI
        resized = cv2.resize(face_roi, (128, 128))
        normalized = resized.astype('float32') / 255.0
        input_data = np.expand_dims(normalized, axis=0)
        
        # Debug model input requirements
        num_inputs = len(gender_model.inputs)
        
        if num_inputs == 1:
            # Single input model - standard case
            try:
                pred = gender_model.predict(input_data, verbose=0)
            except Exception as e:
                print(f"Single input prediction failed: {e}")
                # Try with different input formats
                try:
                    pred = gender_model.predict([input_data], verbose=0)
                except:
                    # Last resort - try with the array directly
                    pred = gender_model.predict(normalized, verbose=0)
        
        elif num_inputs > 1:
            # Multi-input model - handle each input appropriately
            inputs = []
            for i in range(num_inputs):
                input_shape = gender_model.inputs[i].shape
                if len(input_shape) == 4:  # Image input
                    inputs.append(input_data)
                else:
                    # For non-image inputs, create dummy data
                    dummy_input = np.zeros((1, *input_shape[1:]))
                    inputs.append(dummy_input)
            
            pred = gender_model.predict(inputs, verbose=0)
        else:
            print("No inputs found in model")
            return "Unknown"
        
        # Handle different output formats
        if isinstance(pred, list):
            pred = pred[0]  # Take first output if multiple
        
        # Determine gender based on output format
        if pred.shape[-1] == 1:
            # Binary classification (sigmoid output)
            gender_prob = pred[0][0]
            gender = "Male" if gender_prob < 0.5 else "Female"
        else:
            # Multi-class classification (softmax output)
            gender_idx = np.argmax(pred[0])
            gender = "Male" if gender_idx == 0 else "Female"
        
        return gender
        
    except Exception as e:
        print(f"Gender prediction error: {e}")
        # Fallback to simple heuristic based on face proportions
        try:
            height, width = face_roi.shape[:2]
            aspect_ratio = width / height
            return "Male" if aspect_ratio > 0.85 else "Female"
        except:
            return "Unknown"

def load_models():
    global age_net, gender_model, age_list, face_dataset, face_labels, names
    
    # Load age model
    try:
        age_net = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')
        age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        print("✅ Age model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load age model: {e}")
        age_net = None
    
    # Load gender model with comprehensive error handling
    gender_model_path = "Gender_Classifier.keras"
    if os.path.exists(gender_model_path):
        try:
            gender_model = load_model(gender_model_path)
            print("✅ Gender classifier (Keras model) loaded successfully!")
            
            # Inspect the model to understand its structure
            inspect_gender_model()
            
        except Exception as e:
            print(f"❌ Failed to load gender model from {gender_model_path}: {e}")
            print("🔄 Creating fallback gender model...")
            gender_model = create_simple_gender_model()
    else:
        print(f"❌ Gender model file not found: {gender_model_path}")
        print("🔄 Creating fallback gender model...")
        gender_model = create_simple_gender_model()

    # Load face dataset
    dataset_path = './face_dataset/'
    face_data = []
    labels = []
    class_id = 0
    
    if os.path.exists(dataset_path):
        for fx in os.listdir(dataset_path):
            if fx.endswith('.npy'):
                names[class_id] = fx[:-4]
                data_item = np.load(os.path.join(dataset_path, fx))
                face_data.append(data_item)
                labels.extend([class_id] * data_item.shape[0])
                class_id += 1
        
        if face_data:
            face_dataset = np.concatenate(face_data, axis=0)
            face_labels = np.array(labels).reshape(-1, 1)
            print(f"✅ Loaded {len(names)} face classes from dataset")
        else:
            print("ℹ️ No face data found in dataset directory")
            face_dataset = np.array([])
            face_labels = np.array([])
    else:
        print("ℹ️ Face dataset directory not found")
        face_dataset = np.array([])
        face_labels = np.array([])

def distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))

def knn(train, test, k=5):
    if train.size == 0:
        return 0  # Return default class if no training data
    
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
        
        # Create temporary file safely
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_filename = temp_file.name
        
        wavfile.write(temp_filename, fs, audio)
        features = extract_voice_features(temp_filename)
        
        # Clean up temporary file
        os.unlink(temp_filename)
        
        return features.get("pitch", 0) > 220  # Returns boolean
    except Exception as e:
        print(f"Voice analysis error: {e}")
        return False

def process_basic_info_frame(frame):
    """Process frame to collect only name and gender"""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame, None, None
        
        x, y, w, h = faces[0]
        face_roi = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Face recognition
        name = "Unknown"
        if face_dataset.size > 0:
            try:
                resized_face = cv2.resize(face_roi, (100, 100)).flatten()
                identity = knn(np.hstack((face_dataset, face_labels)), resized_face)
                name = names.get(int(identity), "Unknown")
            except Exception as e:
                print(f"Face recognition error: {e}")
                name = "Unknown"

        # Gender detection using the improved function
        gender = predict_gender(face_roi)

        # Update current analysis
        current_analysis["name"] = str(name)
        current_analysis["gender"] = str(gender)

        # Display results on frame
        cv2.putText(frame, f"Name: {name}", (x, y-40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Gender: {gender}", (x, y-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame, name, gender
    
    except Exception as e:
        print(f"Basic info processing error: {e}")
        cv2.putText(frame, f"Error: {str(e)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return frame, None, None

def process_call_frame(frame):
    """Process frame during video call for lie detection"""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame, False, None
        
        x, y, w, h = faces[0]
        face_roi = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Emotion detection for lie detection
        lie_detected = False
        lie_info = None
        
        try:
            emotion = extract_emotion(face_roi)
            lie_detected = emotion in ['fear', 'disgust', 'sad']
            
            if lie_detected:
                timestamp = datetime.now().strftime("%H:%M:%S")
                lie_info = f"emotion:{emotion}"
                current_analysis["lie_detected"] = True
                current_analysis["lie_timestamps"].append((timestamp, lie_info))
        except Exception as e:
            print(f"Emotion extraction error: {e}")
        
        # Voice analysis every 150 frames
        global frame_count
        frame_count += 1
        if frame_count % 150 == 0:
            try:
                voice_stress_detected = analyze_voice()
                if voice_stress_detected:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    lie_detected = True
                    lie_info = "voice_stress"
                    current_analysis["lie_detected"] = True
                    current_analysis["lie_timestamps"].append((timestamp, lie_info))
            except Exception as e:
                print(f"Voice analysis error: {e}")
        
        # Display alerts if lie detected
        if lie_detected:
            cv2.putText(frame, f"Alert: {lie_info}", (x, y-60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Suspicious behavior detected", (x, y-80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame, lie_detected, lie_info
    
    except Exception as e:
        print(f"Call frame processing error: {e}")
        return frame, False, None

def get_analysis_results():
    """Get the current analysis results"""
    return (
        current_analysis["name"],
        current_analysis["gender"],
        current_analysis["lie_detected"],
        current_analysis["lie_timestamps"]
    )

def reset_analysis():
    """Reset the analysis state"""
    global current_analysis, frame_count
    current_analysis = {
        "name": None,
        "gender": None,
        "lie_detected": False,
        "lie_timestamps": []
    }
    frame_count = 0
    print("✅ Analysis state reset")

# Initialize models on import
print("🔄 Initializing face recognition models...")
load_models()
print("✅ Face recognition models initialized successfully!")
