import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread('lena.jpg')
# cv2.imwrite('Written Image.jpg', image)
    

cv2.imshow('Original Image', image)
cv2.waitKey(0)



#GrayScale Image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale Image', gray_image)
cv2.waitKey(0)

#HSV colorspace
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow('HSV Image', hsv_image)
cv2.waitKey(0)

# arithmetic
arithmetic_image = cv2.add(image, 50)
cv2.imshow('Arithmetic Image', arithmetic_image)
cv2.waitKey(0)

# bitwise operations 
bitwise_image = cv2.bitwise_not(image)
cv2.imshow('Bitwise Image', bitwise_image)
cv2.waitKey(0)

#interpolation techniques
nearest_neighbor = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
bilinear = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
bicubic = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
lanczos = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
cv2.imshow('Nearest Neighbor', nearest_neighbor)
cv2.imshow('Bilinear', bilinear)
cv2.imshow('Bicubic', bicubic)
cv2.imshow('Lanczos', lanczos)
cv2.waitKey(0)

# Crop, flip, and rotate
cropped_image = image[100:400, 200:500]
flipped_image = cv2.flip(image, 1)
rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
cv2.imshow('Cropped Image', cropped_image)
# cv2.imwrite('Cropped Image',cropped_image)
cv2.imshow('Flipped Image', flipped_image)
cv2.imshow('Rotated Image', rotated_image)
cv2.waitKey(0)

# contrast and brightness
contrast_brightness_image = cv2.convertScaleAbs(image, alpha=2.0, beta=50)
cv2.imshow('Contrast Brightness Image', contrast_brightness_image)
cv2.waitKey(0)

# template matching technique
template = cv2.imread('lena_tmpl.jpg', cv2.IMREAD_GRAYSCALE)
result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
top_left = max_loc
bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
cv2.imshow('Template Matching', image)
cv2.waitKey(0)

#histogram
histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
plt.plot(histogram)
plt.title('Grayscale Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()

#Gaussian
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow('Blurred Image', blurred_image)
cv2.waitKey(0)

# thresholding 
_, binary_threshold = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
cv2.imshow('Binary Threshold', binary_threshold)
cv2.waitKey(0)

#morphological transformations 
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(binary_threshold, kernel, iterations=1)
dilation = cv2.dilate(binary_threshold, kernel, iterations=1)
opening = cv2.morphologyEx(binary_threshold, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(binary_threshold, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(binary_threshold, cv2.MORPH_GRADIENT, kernel)
tophat = cv2.morphologyEx(binary_threshold, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(binary_threshold, cv2.MORPH_BLACKHAT, kernel)


cv2.imshow('Erosion', erosion)
cv2.imshow('Dilation', dilation)
cv2.imshow('Opening', opening)
cv2.imshow('Closing', closing)
cv2.imshow('Gradient', gradient)
cv2.imshow('Top Hat', tophat)
cv2.imshow('Black Hat', blackhat)
cv2.waitKey(0)

# edge detection
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
# prewitt_x = cv2.filter2D(image, -1, kernel_x)
# prewitt_y = cv2.filter2D(image, -1, kernel_y)
# roberts_x = cv2.filter2D(image, -1, kernel_roberts_x)
# roberts_y = cv2.filter2D(image, -1, kernel_roberts_y)

sobel_mag = cv2.magnitude(sobel_x, sobel_y)
edges = cv2.Canny(image, 100, 200)

laplacian = cv2.Laplacian(image, cv2.CV_64F)


cv2.imshow('Original Image', image)
cv2.imshow('Sobel X', sobel_x)
cv2.imshow('Sobel Y', sobel_y)
cv2.waitKey(0)

# contours 
contours, hierarchy = cv2.findContours(binary_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
cv2.imshow('Contours', image)
cv2.waitKey(0)
#printing metadata
capture=cv2.VideoCapture(0) 
print("CV_CAP_PROP_FRAME_WIDTH: '{}'".format(capture.get(cv2.CAP_PROP_FRAME_WIDTH))) 
print("CV_CAP_PROP_FRAME_HEIGHT : '{}'".format(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))) 
print("CAP_PROP_FPS : '{}'".format(capture.get(cv2.CAP_PROP_FPS))) 
print("CAP_PROP_POS_MSEC : '{}'".format(capture.get(cv2.CAP_PROP_POS_MSEC))) 
print("CAP_PROP_FRAME_COUNT : '{}'".format(capture.get(cv2.CAP_PROP_FRAME_COUNT))) 
print("CAP_PROP_BRIGHTNESS : '{}'".format(capture.get(cv2.CAP_PROP_BRIGHTNESS))) 
print("CAP_PROP_CONTRAST : '{}'".format(capture.get(cv2.CAP_PROP_CONTRAST))) 
print("CAP_PROP_SATURATION : '{}'".format(capture.get(cv2.CAP_PROP_SATURATION))) 
print("CAP_PROP_HUE : '{}'".format(capture.get(cv2.CAP_PROP_HUE))) 
print("CAP_PROP_GAIN : '{}'".format(capture.get(cv2.CAP_PROP_GAIN))) 
print("CAP_PROP_CONVERT_RGB : '{}'".format(capture.get(cv2.CAP_PROP_CONVERT_RGB))) 

capture.release() 
cv2.destroyAllWindows()


cv2.destroyAllWindows()
