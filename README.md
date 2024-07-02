# Human-Scream-Detection-and-analysis
import numpy as np
import librosa
import requests
import json
from geopy.geocoders import Nominatim
from sklearn.externals import joblib

# Load pre-trained scream detection model
model = joblib.load('scream_detection_model.pkl')

# Function to send emergency notification
def send_emergency_notification(location):
    url = "https://api.example.com/send-notification"
    data = {
        "message": "Emergency: Scream detected",
        "location": location
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    return response.status_code

# Function to analyze audio and detect screams
def analyze_audio(file_path):
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

# Function to get location from coordinates
def get_location():
    geolocator = Nominatim(user_agent="scream_detection_app")
    location = geolocator.geocode("Your address or coordinates")
    return location.address if location else "Unknown location"

# Function to detect scream and take action
def detect_scream_and_notify(file_path):
    features = analyze_audio(file_path)
    features = features.reshape(1, -1)
    prediction = model.predict(features)
    
    if prediction == 1:  # Assuming '1' means scream detected
        location = get_location()
        response_code = send_emergency_notification(location)
        if response_code == 200:
            print(f"Emergency notification sent successfully to police station at {location}.")
        else:
            print("Failed to send emergency notification.")
    else:
        print("No scream detected.")

# Example usage
audio_file_path = "path/to/audio/file.wav"
detect_scream_and_notify(audio_file_path)
