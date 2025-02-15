import keras
import cv2
import numpy as np
import argparse

def predict(img):
    # Load the model
    model = keras.models.load_model('pneumoniaClassifier.keras', compile=False)
    model.load_weights('pneumoniaClassifier.keras')
    # Preprocess the image
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (224, 224))
    img = np.stack([img,img,img],axis=-1)
    img = img / 255.0
    # Make predictions
    predictions = model.predict(np.array([img]))
    # print(predictions)
    normal = 0
    pneumonia = 0
    for item in predictions:
      if item[0][0] > 0.8:
        pneumonia += 1
      else:
        normal += 1
    return "Pneumonia" if pneumonia > normal else "Normal"

if __name__ == "__main__":
   
  # Argument parser
  parser = argparse.ArgumentParser(description="Pneumonia X-ray Classifier")
  parser.add_argument("--image_path", type=str, required=True, help="Path to the X-ray image")
  args = parser.parse_args()

  # put your image path here
  img_path = args.image_path
  img = cv2.imread(img_path)

  if img is None:
        print("Error: Unable to read image. Check the file path.")
  else:
      print(predict(img))