#TODO No previous approach worked, but hasn't tested on 'strategical view', test it, that way a preprocessed image can be fed to a neural network
import cv2
import numpy as np

# Load the image
image = cv2.imread('path_to_your_screenshot.png')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Edge detection
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours
for contour in contours:
    # Approximate the contour to a polygon
    polygon = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

    # Check if the polygon has six sides
    if len(polygon) == 6:
        # Check if the shape is roughly convex, as hexagons should be
        if cv2.isContourConvex(polygon):
            # This contour is likely a hexagon - you can now extract it
            # Create a mask for the hexagon
            mask = np.zeros_like(gray)
            cv2.fillPoly(mask, [polygon], 255)

            # Use bitwise AND to extract the hexagon
            hexagon = cv2.bitwise_and(image, image, mask=mask)

