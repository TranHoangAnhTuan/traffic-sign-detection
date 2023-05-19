import cv2
import numpy as np

# Read the input image
img = cv2.imread('C:\working\cs50AI\week5\gtsrb\gtsrb\\7\\00000_00014.ppm', cv2.IMREAD_GRAYSCALE)

# Blur the image to reduce noise
img = cv2.GaussianBlur(img, (5,5), 0)

# Apply thresholding to create a binary image
ret, thresh = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Fit ellipses to the contours
for contour in contours:
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(img, ellipse, (0, 255, 0), 2)

# Display the image with the detected ellipses
cv2.imshow('Detected Shapes', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
