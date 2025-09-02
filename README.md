# IMAGE-TRANSFORMATIONS

## Aim :
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required :
Anaconda - Python 3.7

## Algorithm :
### Step 1 :
Read and convert the image to RGB format using OpenCV and NumPy.
### Step 2 :
Apply translation by shifting the image 50 pixels right and 100 pixels down.
### Step 3 :
Scale the image by a factor of 1.5 in both horizontal and vertical directions.
### Step 4 :
Shear the image using an affine transformation with shear factors.
### Step 5 :
Apply horizontal reflection, rotate the image by 45 degrees, and crop a specific region.

## Program :
```
Developed By: NITHYA D
Register Number: 212223240110
```
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('img.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
rows, cols, _ = image.shape
```
#### i) Image Translation
```
M_translate = np.float32([[1, 0, 50], [0, 1, 100]])  # Translate by (50, 100) pixels
translated_image = cv2.warpAffine(image_rgb, M_translate, (cols, rows))
```
#### ii) Image Scaling
```
scaled_image = cv2.resize(image_rgb, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
```
#### iii) Image shearing
```
M_shear = np.float32([[1, 0.5, 0], [0.5, 1, 0]])  # Shearing matrix
sheared_image = cv2.warpAffine(image_rgb, M_shear, (int(cols * 1.5), int(rows * 1.5)))
```
#### iv) Image Reflection
```
reflected_image = cv2.flip(image_rgb, 1)  # Horizontal flip
```
#### v) Image Rotation
```
M_rotate = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)  # Rotate by 45 degrees
rotated_image = cv2.warpAffine(image_rgb, M_rotate, (cols, rows))
```
#### vi) Image Cropping
```
cropped_image = image_rgb[50:300, 100:400]  # Crop region from the image
```
#### vii) Display all transformations
```
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(translated_image)
plt.title("i) Translated")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(scaled_image)
plt.title("ii) Scaled")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(sheared_image)
plt.title("iii) Sheared")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(reflected_image)
plt.title("iv) Reflected")
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(rotated_image)
plt.title("v) Rotated")
plt.axis('off')

plt.tight_layout()
plt.show()
```
#### viii) Show cropped image separately
```
plt.figure(figsize=(4, 4))
plt.imshow(cropped_image)
plt.title("vi) Cropped")
plt.axis('off')
plt.show()
```

## Output :
<img width="1189" height="649" alt="image" src="https://github.com/user-attachments/assets/5cfcd5e7-d00d-4463-bd7f-2be0f441079b" />

<img width="330" height="277" alt="image" src="https://github.com/user-attachments/assets/dbf8c621-2907-407a-8b64-0d1907210cf1" />

## Result : 
Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
