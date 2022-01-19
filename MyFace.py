cog_key = 'f0e176242299429f87ce3f707f9d8547'
cog_endpoint = 'https://facialanalysis.cognitiveservices.azure.com/'

from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from python_code import faces
import os
import matplotlib.pyplot as plt
# Create a face detection client.
face_client = FaceClient(cog_endpoint, CognitiveServicesCredentials(cog_key))
# Open an image
image_path = os.path.join('CapFrame.jpg')
image_stream = open(image_path, "rb")
# Detect faces
detected_faces = face_client.face.detect_with_stream(image = image_stream)
# Display the faces(code in python_code / faces.py)
faces.show_faces(image_path, detected_faces)

# Open an image
image_path = os.path.join('CapFrame.jpg')
image_stream = open(image_path, "rb")
# Detect faces
detected_faces = face_client.face.detect_with_stream(image = image_stream)
# Display the faces(code in python_code / faces.py)
faces.show_faces(image_path, detected_faces, show_id = True)


# Open an image
image_path = os.path.join('CapFrame.jpg')
image_stream = open(image_path, "rb")
# Detect faces and specified facial attributes
attributes = ['age', 'emotion']
detected_faces = face_client.face.detect_with_stream(image = image_stream, return_face_attributes = attributes)
# Display the faces and attributes(code in python_code / faces.py)
faces.show_face_attributes(image_path, detected_faces)