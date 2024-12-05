import pyrealsense2 as rs
import requests
import numpy as np
import whisper
import sounddevice as sd
from scipy.io.wavfile import write
from pydub import AudioSegment
import cv2
import os
import base64

# Define the server URL
url = "http://127.0.0.1:8000/predict"  # Note the '/predict' endpoint

def record_audio(duration, filename="audio.mp3", sample_rate=44100):
    save_dir = "audio_files"
    os.makedirs(save_dir, exist_ok=True)
    wav_filename = os.path.join(save_dir, "temp.wav")
    mp3_filename = os.path.join(save_dir, filename)
    print(f"Recording for {duration} seconds...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    write(wav_filename, sample_rate, audio_data)
    sound = AudioSegment.from_wav(wav_filename)
    sound.export(mp3_filename, format="mp3")
    print(f"Recording saved as {mp3_filename}")

record_audio(duration=3, filename="audio.mp3")

selected_model = "base"
model = whisper.load_model(selected_model)
print("----------------------------------------------------")
print("Processing speech-to-text")
result = model.transcribe("audio_files/audio.mp3")
print(result["text"])
print("----------------------------------------------------")

# RealSense camera setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

try:
    pipeline.start(config)
    print("Streaming started. Press 'q' to quit.")
    save_dir = "image_files"
    os.makedirs(save_dir, exist_ok=True)
    count = 0
    color_image = None
    depth_image = None

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            print("Error: Could not read frame.")
            continue

        # Convert RealSense frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Normalize depth image for display
        depth_image_display = cv2.convertScaleAbs(depth_image, alpha=0.03)

        # Display images
        cv2.imshow("Color Image", color_image)
        cv2.imshow("Depth Image", depth_image_display)

        count += 1

        # Convert the color frame to PNG format
        _, buffer = cv2.imencode('.png', color_image)

        # Prepare the payload
        files = {
            "sam_type": (None, "sam2.1_hiera_small"),
            "box_threshold": (None, "0.3"),
            "text_threshold": (None, "0.25"),
            "text_prompt": (None, result["text"]),
            "image": ("image.png", buffer.tobytes(), "image/png")
        }

        # Send the POST request
        response = requests.post(url, files=files)

        if response.status_code == 200:
            response_json = response.json()
            print(f"Processed output received for frame {count}")

            # Decode the base64-encoded image
            img_data = base64.b64decode(response_json["output_image"])
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            output_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            # Display the processed output image
            # cv2.imshow("Output Image", output_image)

            # Get the mask
            mask = response_json["mask"]
            mask = np.array(mask)
            mask_image = (mask * 255).astype(np.uint8)  # Scale 0 to 1 values to 0 to 255
            # Display the mask image
            cv2.imshow("Mask Image", mask_image)
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if count == 10:
            print("Break the loop")
            break

finally:
    pipeline.stop()

    # Save the last captured color and depth images
    color_image_path = os.path.join(save_dir, "color_image.png")
    depth_image_path = os.path.join(save_dir, "depth_image.png")
    output_image_path = os.path.join(save_dir, "output_image.png")
    mask_image_path = os.path.join(save_dir, "mask_image.png")

    cv2.imwrite(color_image_path, color_image)
    cv2.imwrite(depth_image_path, depth_image)
    cv2.imwrite(output_image_path, output_image)
    cv2.imwrite(mask_image_path, mask_image)

    print(f"Color image saved as {color_image_path}")
    print(f"Depth image saved as {depth_image_path}")
    print(f"Output image saved as {output_image_path}")
    print(f"Mask image saved as {mask_image_path}")

    cv2.destroyAllWindows()
