from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", required=True,
# 	help="path to trained model model")
# ap.add_argument("-l", "--labelbin", required=True,
# 	help="path to label binarizer")
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# args = vars(ap.parse_args())


# Random stuff
# im = cv2.imread('/home/sidharth/Downloads/ezgif.com-gif-maker.jpg')
# cv2.imshow('Image', im )
# cv2.waitKey(0)


image = cv2.imread('21.2 Web_Classification_models/NO CODE3.png')
output = image.copy()

# pre-process the image for classification
image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

print("[INFO] loading network...")
model = load_model("21.2 Web_Classification_models/webcl.model")
lb = pickle.loads(open('21.2 Web_Classification_models/lb.pickle', "rb").read())

# classify the input image
print("[INFO] classifying image...")
proba = model.predict(image)[0]
# print(proba
idx = np.argmax(proba)
label = lb.classes_[idx]
print(label)
if label !='CODE': #Only to check if image is CODE or NOT.
    label = 'NO CODE'
filename = '21.2 Web_Classification_models/NO CODE3.png'['21.2 Web_Classification_models/NO CODE3.png'.rfind(os.path.sep) + 1:]
# correct = "correct" if filename.rfind(label) != -1 else "incorrect"

# build the label and draw the label on the image
# label = "{}: {:.2f}% ({})".format(label, proba[idx] * 100, correct)
label = "{}: {:.2f}%".format(label, proba[idx] * 100)
output = imutils.resize(output, width=500, height=700)
cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2)

# show the output image
print("[INFO] {}".format(label))
cv2.imshow("Output", output)
cv2.waitKey(0)
