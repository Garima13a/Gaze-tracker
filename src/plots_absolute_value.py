import csv
import cv2
import PIL
import os
import pandas as pd
import numpy  as np
from glob import glob
import plotly.express as px
from plotly import subplots
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import plotly.graph_objects as go
from matplotlib.patches import Circle, Rectangle

# 67,68 left eye
# 267, 268 

#725, 726 right eye
#527,528

face = '/home/garima/Desktop/Gaze_Tracking/mediapipe_browser/csv_ploter/csv_stable/original_face_landmarks.csv'
cap= cv2.VideoCapture('/home/garima/Desktop/Gaze_Tracking/dataset/MP4Stable/original_stable.mp4')
img = '/home/garima/Desktop/Gaze_Tracking/dataset/dataset_frames/original_frames/frame2.png'

vn = face.split('/')[-1][:8]
readcsv = pd.read_csv(face, header= None)
eyes_df = pd.DataFrame(readcsv)

ret, frame = cap.read()
im = frame
im.shape

#helper function to show image
def show_image(img, x1,x2,y1,y2, i):
    image_name = img
    im = plt.imread(image_name)
    implot = plt.imshow(im)
    plt.scatter(x=[x1,x2], y=[y1,y2], c='r', s=20)
#     plt.savefig('/home/garima/Desktop/tmp/t' + str(i) + '.png')
    plt.show()

im = plt.imread(img)
i=0
# left_a_x1 = readcsv['x3'][3]*im.shape[1]
# left_a_y1 =  readcsv['y3'][3]*im.shape[0]

# left_b_x1 = readcsv['x1'][3]*im.shape[1]
# left_b_y1 =  readcsv['y1'][3]*im.shape[0]

# right_b_x2 = readcsv['x6'][3]*im.shape[1]
# right_b_y2 =  readcsv['y6'][3]*im.shape[0]

# right_a_x2 = readcsv['x6'][3]*im.shape[1]
# right_a_y2 =  readcsv['y6'][3]*im.shape[0]


x1 = readcsv[267][3]*im.shape[1]
y1 =  readcsv[268][3]*im.shape[0]


x2 =readcsv[527][3]*im.shape[1]
y2 = readcsv[528][3]*im.shape[0]


# x1 = left_b_x1
# y1 = left_b_y1

# x2 = right_b_x2
# y2 = right_b_y2

show_image(img, x1,x2,y1,y2, i)

img = '/home/garima/Desktop/Gaze_Tracking/dataset/dataset_frames/original_frames/frame2.png'
im = plt.imread(img)
i=0
left_a_x1 = readcsv[67][3]*im.shape[1]
left_a_y1 =  readcsv[68][3]*im.shape[0]

left_b_x1 = readcsv[267][3]*im.shape[1]
left_b_y1 =  readcsv[268][3]*im.shape[0]

right_b_x2 = readcsv[527][3]*im.shape[1]
right_b_y2 =  readcsv[528][3]*im.shape[0]

right_a_x2 = readcsv[725][3]*im.shape[1]
right_a_y2 =  readcsv[726][3]*im.shape[0]


# x1 = readcsv['x8'][3]*im.shape[1]
# y1 =  readcsv['y8'][3]*im.shape[0]


# x2 =readcsv['x9'][3]*im.shape[1]
# y2 = readcsv['y9'][3]*im.shape[0]


x1 = left_a_x1
y1 = left_a_y1

x2 = right_a_x2
y2 = right_a_y2

show_image(img, x1,x2,y1,y2, i)

left_a_x1 = readcsv[67]*im.shape[1]
left_a_y1 =  readcsv[68]*im.shape[0]

left_b_x1 = readcsv[267]*im.shape[1]
left_b_y1 =  readcsv[268]*im.shape[0]

right_b_x2 = readcsv[527]*im.shape[1]
right_b_y2 =  readcsv[528]*im.shape[0]

right_a_x2 = readcsv[725]*im.shape[1]
right_a_y2 =  readcsv[726]*im.shape[0]

val1 = abs(left_b_x1 -left_a_x1)
val2 = abs(left_a_y1 -left_b_y1)

val3 = abs(right_b_x2 -right_a_x2)
val4 = abs(right_a_y2 -right_b_y2)


count1 = val1/11.7 # Left hori
count2 = val2/11.7 #Left ver

count3 = val3/11.7 # Right Hori
count4 = val4/11.7 # Right Ver
count4

# vn = "Absolute"
fig = go.Figure()

fig.add_trace(go.Scatter(
        x=eyes_df[0], 
        y=count1, 
        mode='lines', 
        name='Left Hor', 
        line=dict(color='gold'))
    )
fig.add_trace(go.Scatter(
        x=eyes_df[0], 
        y=count2, 
        mode='lines', 
        name='Left Ver', 
        line=dict(color='purple'))
    )
fig.add_trace(go.Scatter(
        x=eyes_df[0], 
        y=count3, 
        mode='lines', 
        name='Right Hor', 
        line=dict(color='red'))
    )
fig.add_trace(go.Scatter(
        x=eyes_df[0], 
        y=count4, 
        mode='lines', 
        name='Right Ver', 
        line=dict(color='blue'))
    )

fig.update_layout(
        title       = vn,
        yaxis       = dict(range=[-0.01,0.01]),
        xaxis_title = r'Time (micro secs)',
        yaxis_title = 'Absolute value(mm)',
        legend      = dict(
            orientation = "h",
            yanchor     = "bottom",
            y           = 1.02,
            xanchor     = "right",
            x           = 1
        )
    )

fig.write_html(f"/home/garima/Desktop/Gaze_Tracking/mediapipe_browser/csv_ploter/html_abs_final/{vn}.html")