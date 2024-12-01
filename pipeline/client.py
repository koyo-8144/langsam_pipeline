# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# import requests

# response = requests.post("http://127.0.0.1:8000/predict", json={"input": 4.0})
# print(f"Status: {response.status_code}\nResponse:\n {response.text}")


import requests


# Define the server URL
url = "http://127.0.0.1:8000/predict"  # Note the '/predict' endpoint

# Define the parameters
sam_type = "sam2.1_hiera_small"
box_threshold = 0.3
text_threshold = 0.25
text_prompt = "extract a car"



# Open the image file (replace with your image path)
image_path = "/home/koyo/lang-segment-anything/lang_sam/images/car2.jpeg"
with open(image_path, "rb") as image_file:
    # Prepare the payload (including the image and other parameters)
    files = {
        "sam_type": (None, sam_type),
        "box_threshold": (None, str(box_threshold)),
        "text_threshold": (None, str(text_threshold)),
        "text_prompt": (None, text_prompt),
        "image": ("image.jpg", image_file, "image/jpeg")  # Send the image as a file
    }

    print("files ", files)
    # Send the POST request
    response = requests.post(url, files=files)

# Check if the response is successful
if response.status_code == 200:
    #print("Output image: ", response.content)

    # Save the output image
    with open("output_image.png", "wb") as output_file:
        output_file.write(response.content)
    print("Output image saved as output_image.png")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
