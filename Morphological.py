import cv2
print(cv2.__version__)
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('blur1.webp', 0)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')
plt.show()
kernel = np.ones((5, 5), np.uint8)
img_erosion = cv2.erode(img, kernel, iterations=1)

plt.imshow(img_erosion, cmap='gray')
plt.title("After Erosion")
plt.axis('off')

plt.show()
img_dilation = cv2.dilate(img, kernel, iterations=1)

plt.imshow(img_dilation, cmap='gray')
plt.title("After Dilation")
plt.axis('off')
plt.show()
import cv2

# Read the image.
img = cv2.imread('blur1.webp')

# Apply bilateral filter with d = 15, 
# sigmaColor = sigmaSpace = 75.
bilateral = cv2.bilateralFilter(img, 15, 75, 75)

# Save the output.
cv2.imwrite('blur1.webp', bilateral)

# Importing required libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Reading the noisy image from file
img = cv2.imread('blur1.webp')

# Applying denoising filter
dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)

# Displaying original and denoised images
plt.subplot(121), plt.imshow(img), plt.title('Original Image')
plt.subplot(122), plt.imshow(dst), plt.title('Denoised Image')
plt.show()

import cv2
import numpy as np

cap = cv2.VideoCapture(0)  # Start webcam

while True:
    _, frame = cap.read()
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range for blue color in HSV
    lower_blue = np.array([60, 35, 140])
    upper_blue = np.array([180, 255, 255])

    # Create mask
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Filter the blue region
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Show frames
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Blue Mask', mask)
    cv2.imshow('Blue Filtered Result', result)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit the loop when 'q' is pressed

cap.release()
cv2.destroyAllWindows()
print("Exiting...")  # Confirm exit