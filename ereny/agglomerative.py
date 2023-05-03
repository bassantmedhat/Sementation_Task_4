import cv2
import numpy as np
import matplotlib.pyplot as plt
clusters_list = []
cluster = {}
centers = {}

def calculate_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def clusters_average_distance(cluster1, cluster2):
   
    cluster1_center = np.average(cluster1)
    cluster2_center = np.average(cluster2)
    return calculate_distance(cluster1_center, cluster2_center) 

def initial_clusters(image_clusters):
  
    global initial_k
    groups = {}
    cluster_color = int(256 / initial_k)
    for i in range(initial_k):
        color = i * cluster_color
        groups[(color, color, color)] = []
    for i, p in enumerate(image_clusters):
        go = min(groups.keys(), key=lambda c: np.sqrt(np.sum((p - c) ** 2)))
        groups[go].append(p)
    return [group for group in groups.values() if len(group) > 0]

def get_cluster_center( point):
    global cluster
    point_cluster_num = cluster[tuple(point)]
    center = centers[point_cluster_num]
    return center

def get_clusters(image_clusters):
    global clusters_list
    clusters_list = initial_clusters(image_clusters)

    while len(clusters_list) > clusters_number:
        cluster1, cluster2 = min(
            [(c1, c2) for i, c1 in enumerate(clusters_list) for c2 in clusters_list[:i]],
            key=lambda c: clusters_average_distance(c[0], c[1]))

        clusters_list = [cluster_itr for cluster_itr in clusters_list if cluster_itr != cluster1 and cluster_itr != cluster2]

        merged_cluster = cluster1 + cluster2

        clusters_list.append(merged_cluster)

    global cluster 
    for cl_num, cl in enumerate(clusters_list):
        for point in cl:
            cluster[tuple(point)] = cl_num

    global centers 
    for cl_num, cl in enumerate(clusters_list):
        centers[cl_num] = np.average(cl, axis=0)

def apply_agglomerative_clustering(image, number_of_clusters,initial_number_of_clusters):
    global clusters_number
    global initial_k

    clusters_number = number_of_clusters
    initial_k = initial_number_of_clusters 
    flattened_image = np.copy(image.reshape((-1, 3)))

    get_clusters(flattened_image)
    output_image = []
    for row in image:
        rows = []
        for col in row:
            rows.append(get_cluster_center(list(col)))
        output_image.append(rows)    
    output_image = np.array(output_image, np.uint8)
    return output_image    


img = cv2.imread("1.jpg")
segmented_image = apply_agglomerative_clustering(img,15,30)
# cv2.imshow('Original Image', img)
# cv2.imshow('Segmented Image', segmented_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax = axes.ravel()
ax[0].imshow(img)
ax[0].set_title('Original Image')
ax[0].set_axis_off()
ax[1].imshow(segmented_image ,cmap = "gray")
ax[1].set_title('segmented Image')
ax[1].set_axis_off()
plt.show()
