
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")
from skimage import data
from skimage.feature import match_template
import glob
import cv2
from sklearn.cluster import KMeans
import tensorflow as tf

# slices image into two vertical halves
def sliced(img):
#     print(img.shape)
    height, width, channel = img.shape
    width_cutoff = width // 2
    s1 = img[:, :width_cutoff]
    s2 = img[:, width_cutoff:]
    return s1, s2

#find center of the iris by masking
def find_mask(path):
    sample = path
    image = cv2.imread(sample)
    sl1, sl2 = sliced(image)

    lower_red = np.array([220,0,0])  # BGR-code of your lowest red
    upper_red = np.array([255,10,10])   # BGR-code of your highest red 
    mask1 = cv2.inRange(sl1, lower_red, upper_red)
    mask2 = cv2.inRange(sl2, lower_red, upper_red)
    #get all non zero values
    coord1=cv2.findNonZero(mask1)
    coord2=cv2.findNonZero(mask2)
    return coord1, coord2

#apply kmeans and get center of clusters
def get_center(coord1, coord2):
    X1 = np.array(coord1)
    X2 = np.array(coord2)
    coord1 = coord1[:, 0]
    coord2 = coord2[:, 0]
    kmeans1 =KMeans(n_clusters=1, random_state=0).fit(coord1)
    kmeans2 =KMeans(n_clusters=1, random_state=0).fit(coord2)
    cntr1 = kmeans1.cluster_centers_
    cntr2 = kmeans2.cluster_centers_
    return cntr1,cntr2

#helper function to show image
def show_image(img, x1,x2,y1,y2):
    image_name = img
    im = plt.imread(image_name)
    sl1,sl2 = sliced(im)
    f, (ax1,ax2) = plt.subplots(1,2, figsize=(10,8)) 
    ax1.imshow(sl1)
    ax1.scatter(x=[x1], y=[y1], c='r', s=20)
    ax2.imshow(sl2)
    ax2.scatter(x=[x2], y=[y2], c='r', s=20)

# read each file and add it's center to val array
file_name = glob.glob("/home/garima/Desktop/Gaze_Tracking/frames/*.png")
val1 = []
val2 = []
for i in range(len(file_name)):
    coord1, coord2 = find_mask(file_name[i])
    cen1, cen2  = get_center(coord1, coord2)
    val1.append(cen1)
    val2.append(cen2)

#extract x and y values of left and right eye
xs1 = []
xs2 = []
ys1 = []
ys2 = []
half1 = []
half2 = []

for i in range(len(val1)):
    half1.append(val1[i][0])
for k in range(len(val2)):
    half2.append(val2[k][0])
for x in range(len(half1)):
    xs1.append(half1[x][0])
    ys1.append(half1[x][1])
for y in range(len(half2)):
    xs2.append(half2[y][0])
    ys2.append(half2[y][1])

#plot all
plt.plot(xs2)
plt.xlabel("Number of frames")
plt.ylabel("left eye x axis movement")

plt.plot(xs1)
plt.xlabel("Number of frames")
plt.ylabel("right eye x axis movement")

plt.plot(ys2)
plt.xlabel("Number of frames")
plt.ylabel("left eye y axis movement")

plt.plot(ys1)
plt.xlabel("Number of frames")
plt.ylabel("right eye y axis movement")

path1 = '../Gaze_Tracking/frames_4/frame175.png'
coord1, coord2 = find_mask(path1)
cntr1, cntr2 = get_center(coord1, coord2)
show_image(path1, cntr1[0][0], cntr2[0][0], cntr1[0][1],cntr2[0][1])

f, (ax1,ax2,ax3, ax4) = plt.subplots(4,1, figsize=(15,15)) 

ax1.plot(xs2)
ax1.set(ylabel='Left eye horizontal')

ax2.plot(xs1)
ax2.set(ylabel='Right eye horizontal')

ax3.plot(ys2)
ax3.set(ylabel='Left eye vertical')

ax4.plot(ys1)
ax4.set(xlabel = 'Number of frames',ylabel='Right eye vertical')

plt.savefig('./graphs/hello.png')