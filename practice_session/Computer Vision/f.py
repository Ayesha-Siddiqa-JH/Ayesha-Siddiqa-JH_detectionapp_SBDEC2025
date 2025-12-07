#CODE 10

import cv2

img = cv2.imread("images.jpg", 0)

edges = cv2.Canny(img, 100, 200)

cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()



#CODE 11

import cv2

# Load pre-trained classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

img = cv2.imread("images.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow("Face Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#CODE 12

import cv2

img = cv2.imread("images.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

cv2.imshow("Contours", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#CODE 13

import cv2

img = cv2.imread("images.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define blue color range
lower_blue = (100, 150, 0)
upper_blue = (140, 255, 255)

mask = cv2.inRange(hsv, lower_blue, upper_blue)
result = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow("Original", img)
cv2.imshow("Mask", mask)
cv2.imshow("Filtered", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
#CODE 14

import cv2
import numpy as np

img = cv2.imread("images.jpg")
mask = np.zeros(img.shape[:2], np.uint8)

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

rect = (50, 50, 400, 500)  # ROI

cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
result = img * mask2[:, :, np.newaxis]

cv2.imshow("Original", img)
cv2.imshow("Foreground Extracted", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
#CODE 15

import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Blue color range
    lower_blue = (100, 150, 0)
    upper_blue = (140, 255, 255)

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Tracked", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#CODE 16

import cv2
import numpy as np

img = cv2.imread("images.jpg", 0)

_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

kernel = np.ones((5, 5), np.uint8)

erosion = cv2.erode(thresh, kernel, iterations=1)
dilation = cv2.dilate(thresh, kernel, iterations=1)

cv2.imshow("Original", thresh)
cv2.imshow("Erosion", erosion)
cv2.imshow("Dilation", dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()
