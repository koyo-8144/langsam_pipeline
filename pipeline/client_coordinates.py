import cv2
import requests
import numpy as np

import whisper
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from pydub import AudioSegment
# from playsound import playsound
import os
import base64


# Define the server URL
url = "http://127.0.0.1:8000/predict"  # Note the '/predict' endpoint


def record_audio(duration, filename="audio.mp3", sample_rate=44100):
    # Define the directory to save audio files
    save_dir = "audio_files"
    
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Full paths for the audio files
    wav_filename = os.path.join(save_dir, "temp.wav")
    mp3_filename = os.path.join(save_dir, filename)
    
    print(f"Recording for {duration} seconds...")
    # Record audio using sounddevice
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    
    # Save as a temporary WAV file
    write(wav_filename, sample_rate, audio_data)
    
    # Convert WAV to MP3
    sound = AudioSegment.from_wav(wav_filename)
    sound.export(mp3_filename, format="mp3")
    
    print(f"Recording saved as {mp3_filename}")

# Record your voice for 3 seconds and save as "audio.mp3"
record_audio(duration=3, filename="audio.mp3")

# # Play the recorded audio
# print("Playing recorded audio...")
# playsound("audio.mp3")

selected_model = "base"
# tiny: 1GB
# base: 1GB
# small: 2GB -> out of memory sometimes
# medium: 5GB -> out of memoory
# large: 10GB -> out of memoory
# turbo: 6GB  -> out of memoory
model = whisper.load_model(selected_model)

print("Processing speech-to-text")
result = model.transcribe("audio_files/audio.mp3")
print(result["text"])

# Define the parameters
sam_type = "sam2.1_hiera_small"
box_threshold = 0.3
text_threshold = 0.25
# text_prompt = "extract a banana"
text_prompt = result["text"]

# Start capturing video from the webcam
#cap = cv2.VideoCapture(6) #v4l2-ctl --list-devices
cap = cv2.VideoCapture(0) #v4l2-ctl --list-devices


if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Capture a single frame from the webcam
ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame.")
    cap.release()
    exit()

# Convert the frame to PNG format
_, buffer = cv2.imencode('.png', frame)

# Prepare the payload
files = {
    "sam_type": (None, sam_type),
    "box_threshold": (None, str(box_threshold)),
    "text_threshold": (None, str(text_threshold)),
    "text_prompt": (None, text_prompt),
    "image": ("image.png", buffer.tobytes(), "image/png")
}


# Send the POST request
response = requests.post(url, files=files)

# Check if the response is successful
if response.status_code == 200:
    # Parse the JSON response
    response_json = response.json()

    # Get the object coordinates from the response
    object_coordinates = response_json["object_coordinates"]
    object_coordinates = np.array(object_coordinates)
    print(f"Object coordinates: {object_coordinates}")

    # Separate x and y coordinates
    x_coords = object_coordinates[:, 0]
    y_coords = object_coordinates[:, 1]

    # Calculate the center (centroid) of the coordinates
    center_x = np.mean(x_coords)
    center_y = np.mean(y_coords)

    # The center of the object
    center = (center_x, center_y)

    print("Center of the object:", center)
else:
    print(f"Error: {response.status_code}")
    print(response.text)

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()