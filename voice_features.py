import librosa
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile

def record_audio(duration=5, sr=22050):
    print(f"🎙️ Recording voice for {duration} seconds...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    wav.write(temp_wav.name, sr, audio)
    return temp_wav.name

def extract_voice_features(audio_path):
    y, sr = librosa.load(audio_path)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    pitch, _ = librosa.piptrack(y=y, sr=sr)
    pitch_mean = pitch[pitch > 0].mean() if np.any(pitch > 0) else 0

    return {
        "mfcc": mfccs,
        "pitch": pitch_mean
    }
