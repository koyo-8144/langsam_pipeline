from io import BytesIO

import litserve as ls
import numpy as np
# from fastapi import Response, UploadFile
from fastapi import FastAPI, Response, UploadFile, Form
from fastapi.responses import JSONResponse
import base64
from PIL import Image

from lang_sam import LangSAM
from lang_sam.utils import draw_image
# from utils import draw_image


PORT = 8000

app = FastAPI()


class LangSAMAPI(ls.LitAPI):
    def setup(self, device: str) -> None:
        """Initialize or load the LangSAM model."""
        self.model = LangSAM(sam_type="sam2.1_hiera_small")
        print("LangSAM model initialized.")
        # print("model ", self.model)
        

    # 1
    def decode_request(self, request) -> dict:

        print("decode_request")
        """Decode the incoming request to extract parameters and image bytes.

        Assumes the request is sent as multipart/form-data with fields:
        - sam_type: str
        - box_threshold: float
        - text_threshold: float
        - text_prompt: str
        - image: UploadFile
        """
        # Extract form data
        sam_type = request.get("sam_type")
        box_threshold = float(request.get("box_threshold", 0.3))
        text_threshold = float(request.get("text_threshold", 0.25))
        text_prompt = request.get("text_prompt", "")


        # Extract image file
        image_file: UploadFile = request.get("image")
        if image_file is None:
            raise ValueError("No image file provided in the request.")

        # print("image_file: ", image_file)
        # UploadFile(filename='image.jpg', size=7467, 
        # headers=Headers({'content-disposition': 'form-data; name="image"; 
        # filename="image.jpg"', 'content-type': 'image/jpeg'}))

        image_bytes = image_file.file.read()

        return {
            "sam_type": sam_type,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
            "image_bytes": image_bytes,
            "text_prompt": text_prompt,
        }

    # 2
    def predict(self, inputs: dict) -> dict:
        print("predict")

        """Perform prediction using the LangSAM model.

        Yields:
            dict: Contains the processed output image.
        """
        print("Starting prediction with parameters:")
        print(
            f"sam_type: {inputs['sam_type']}, \
                box_threshold: {inputs['box_threshold']}, \
                text_threshold: {inputs['text_threshold']}, \
                text_prompt: {inputs['text_prompt']}"
        )

        if inputs["sam_type"] != self.model.sam_type:
            print(f"Updating SAM model type to {inputs['sam_type']}")
            self.model.sam.build_model(inputs["sam_type"])

        try:
            image_pil = Image.open(BytesIO(inputs["image_bytes"])).convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid image data: {e}")
        
        # Get image dimensions
        width, height = image_pil.size
        print(f"Image dimensions: Width = {width}, Height = {height}") # Width = 640, Height = 480

        results = self.model.predict(
            images_pil=[image_pil],
            texts_prompt=[inputs["text_prompt"]],
            box_threshold=inputs["box_threshold"],
            text_threshold=inputs["text_threshold"],
        )
        results = results[0]

        if not len(results["masks"]):
            print("No masks detected. Returning original image.")
            return {"output_image": image_pil}
        
        # Draw results on the image
        image_array = np.asarray(image_pil)

        output_image = draw_image(
            image_array,
            results["masks"],
            results["boxes"],
            results["scores"],
            results["labels"],
        )
        output_image = Image.fromarray(np.uint8(output_image)).convert("RGB")

        masks = results["masks"]
        # Remove the first dimension to simplify (1, height, width) -> (height, width)
        mask = masks[0]
        # Find the coordinates where the mask is 1 (object detected)
        object_coordinates = np.argwhere(mask == 1)
        print("object coordinates: ", object_coordinates)


        # return {"output_image": output_image}
        return {"output_image": output_image, "object_coordinates": object_coordinates.tolist()}

    # 3
    def encode_response(self, output: dict) -> Response:
        print("encode_response")

        """Encode the prediction result into an HTTP response.

        Returns:
            Response: Contains the processed image in PNG format.
        """
        try:
            image = output["output_image"]
            coordinates = output["object_coordinates"]

            # buffer = BytesIO()
            # image.save(buffer, format="PNG")
            # buffer.seek(0)

            # return Response(content=buffer.getvalue(), media_type="image/png")

            # Ensure the image is a byte representation (for PIL Image objects)
            if isinstance(image, Image.Image):  # Check if image is PIL.Image
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                buffer.seek(0)
                image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            elif isinstance(image, bytes):  # If it's already a byte object
                image_data = base64.b64encode(image).decode('utf-8')
            else:
                raise ValueError("Invalid image format.")

            response = {
                "output_image": image_data,
                "object_coordinates": coordinates
            }

            return JSONResponse(content=response)
        
        except StopIteration:
            raise ValueError("No output generated by the prediction.")


lit_api = LangSAMAPI()
server = ls.LitServer(lit_api)

# Define the POST endpoint for prediction
@app.post("/predict")
async def predict(
    sam_type: str = Form(...),
    box_threshold: float = Form(0.3),
    text_threshold: float = Form(0.25),
    text_prompt: str = Form(...),
    image: UploadFile = Form(...)
):
    inputs = {
        "sam_type": sam_type,
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
        "text_prompt": text_prompt,
        "image_bytes": await image.read(),
    }

    # Use LangSAMAPI to make a prediction
    output = lit_api.predict(inputs)
    return lit_api.encode_response(output)


if __name__ == "__main__":
    print(f"Starting LitServe and Gradio server on port {PORT}...")
    server.run(port=PORT)
