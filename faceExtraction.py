from deepface import DeepFace
import cv2
import numpy as np
import io

def extract_and_save_face(input_image_buffer):
    try:
        # Convert BytesIO buffer to numpy array
        file_bytes = np.frombuffer(input_image_buffer.read(), dtype=np.uint8)

        # Decode the image from the buffer
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Detect faces in the image
        detected_face = DeepFace.detectFace(img, enforce_detection=False, detector_backend='opencv')

        # Check if a face was detected
        if detected_face.size == 0:
            return None

        # Convert the detected face from float to uint8 (standard image format)
        detected_face = (detected_face * 255).astype(np.uint8)

        # Convert the processed image to RGB
        processed_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2RGB)

        # Encode the face image to buffer
        _, buffer = cv2.imencode('.jpg', processed_face)
        face_buffer = io.BytesIO(buffer)

        return face_buffer
    except Exception as e:
        print(f"Error in extract_and_save_face: {e}")
        return None

# Example usage with a BytesIO object
# with open("inputImage.jpg", "rb") as file:
#     face_buffer = extract_and_save_face(io.BytesIO(file.read()))
#     if face_buffer:
#         # Process the extracted face