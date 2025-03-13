import requests
import numpy as np
from cv2 import imread
from scipy.optimize import differential_evolution
from PIL import Image
import io
import sys 



def main():
    path = sys.argv[1]
    image = imread(path)
    print(call_model(image))

MODEL = "http://0.0.0.0:5000/model/predict"

def call_model(image_array):
    # Convert numpy array to bytes
    img_pil = Image.fromarray(np.uint8(image_array))
    
    # Ensure 64x64 size
    if img_pil.size != (64, 64):
        img_pil = img_pil.resize((64, 64))
    
    # Convert to PNG bytes
    buffer = io.BytesIO()
    img_pil.save(buffer, format='PNG')
    buffer.seek(0)
    
    # Send to API
    files = {'image': ('image.png', buffer, 'image/png')}
    response = requests.post(MODEL, files=files)
    
    return response.json()


# Using the special variable 
# __name__
if __name__=="__main__":
    main()
