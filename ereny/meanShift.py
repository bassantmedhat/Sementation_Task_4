import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('im.jpg')

# Define the window size and the stopping criteria for mean shift
window_size = 30
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)

# Reshape the image into a 2D array of pixels
img_2d = img.reshape(-1, 3)

# Define the mean shift function
def mean_shift(data, window_size, criteria):
    num_points, num_features = data.shape
    visited = np.zeros(num_points, dtype=bool)
    labels = -1 * np.ones(num_points, dtype=int)
    label_count = 0
    
    for i in range(num_points):
        if visited[i]:
            continue
        
        center = data[i]
        while True:
            # Find all points within the window centered on the current point
            in_window = np.linalg.norm(data - center, axis=1) < window_size
            
            # Calculate the mean of the points within the window
            new_center = np.mean(data[in_window], axis=0)
            
            # If the center has converged, assign labels to all points in the window
            if np.linalg.norm(new_center - center) < criteria[1]:
                labels[in_window] = label_count
                visited[in_window] = True
                label_count += 1
                break
            
            center = new_center
    
    return labels

# Apply mean shift clustering to the image data
labels = mean_shift(img_2d, window_size, criteria)

# Reshape the cluster labels into the shape of the original image
labels = labels.reshape(img.shape[:2])

# Create a new image where each pixel is assigned the color of its cluster centroid
new_img = np.zeros_like(img)
for i in range(np.max(labels)+1):
    new_img[labels == i] = np.mean(img[labels == i], axis=0)

# Display the original image and the segmented image side by side
cv2.imshow('Original Image', img)
cv2.imshow('Segmented Image', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
