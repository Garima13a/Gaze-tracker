import csv
import cv2
import PIL
import os
import numpy as np
import pandas as pd
from glob import glob
from args import parser
from plotly import subplots
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import plotly.graph_objects as go
from matplotlib.patches import Circle, Rectangle

args    = parser.parse_args()

# 67,68 left eye
# 267, 269 

#725, 726 right eye
#527,529

csvs_path = args.csvs_eye
csvs = glob(csvs_path + '/*__iris_landmarks.csv')
face_csvs= glob(csvs_path + '/*__face_landmarks.csv')

readcsv = pd.read_csv(face, header = None)
eyes_df = pd.DataFrame(readcsv)

#helper function to show image
def show_image(img, x1,x2,y1,y2, i):
    image_name = img
    im = plt.imread(image_name)
    implot = plt.imshow(im)
    plt.scatter(x=[x1,x2], y=[y1,y2], c='r', s=20)
    plt.show()

def mean_sub(df):
    return df - df.mean()

def get_eyelid(face_csv):
    face_csv1 = pd.read_csv(face_csv, header = None)

    left1_x = mean_sub(pd.DataFrame(face_csv1[67]))
    left1_y = mean_sub(pd.DataFrame(face_csv1[68]))

    left2_x = mean_sub(pd.DataFrame(face_csv1[267]))
    left2_y = mean_sub(pd.DataFrame(face_csv1[268]))

    right1_x =  mean_sub(pd.DataFrame(face_csv1[725]))
    right1_y=  mean_sub(pd.DataFrame(face_csv1[726]))

    right2_x =  mean_sub(pd.DataFrame(face_csv1[527]))
    right2_y =  mean_sub(pd.DataFrame(face_csv1[528]))
    
    return left1_x,left1_y, left2_x,left2_y,right1_x,right1_y, right2_x,right2_y


def get_htmls(csv,face_csv):
    for (csv,face_csv) in zip(csvs,face_csvs):
            print(csv[68:79])
            print(face_csv)
            
            left1_x,left1_y, left2_x,left2_y,right1_x,right1_y, right2_x,right2_y = get_eyelid(face_csv)
            
            df = pd.read_csv(csv)
            for k in ['x0','x5','y0', 'y5']:
                df[k] = mean_sub(df[k])

            vn = csv.split("__")[0].replace("csv/","")

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df['time'], 
                y=df['x0'], 
                mode='lines', 
                name='left hor ', 
                line=dict(color='firebrick'))
            )
            #for left eyelid corner x axis
            fig.add_trace(go.Scatter(
                x=df['time'], 
                y=left1_x[67], 
                mode='lines', 
                name='Left1_x', 
                line=dict(color='gold'))
            )

            fig.add_trace(go.Scatter(
                x=df['time'], 
                y=df['x5'], 
                mode='lines', 
                name='right hor', 
                line=dict(color='blue')
            ))
            #for right eyelid corner x axis
            fig.add_trace(go.Scatter(
                x=df['time'], 
                y=right1_x[725], 
                mode='lines', 
                name='right1_x', 
                line=dict(color='chocolate')
            ))
            
            fig.add_trace(go.Scatter(
                x=df['time'], 
                y=df['y0'], 
                mode='lines', 
                name='left ver', 
                line=dict(color='green')   )
            )
            
            #for left eyelid corner y axis
            fig.add_trace(go.Scatter(
                x=df['time'], 
                y=left1_y[68], 
                mode='lines', 
                name='left1_y', 
                line=dict(color='mediumorchid')   )
            )
            
            fig.add_trace(go.Scatter(
                x=df['time'], 
                y=df['y5'], 
                mode='lines', 
                name='right ver', 
                line=dict(color='yellow'))
            )
            
            #for right eyelid corner y axis
            fig.add_trace(go.Scatter(
                x=df['time'], 
                y=right1_y[726], 
                mode='lines', 
                name='right1_y', 
                line=dict(color='darkolivegreen'))
            )


            fig.update_layout(
                title       = vn,
                yaxis       = dict(range=[-0.01,0.01]),
                xaxis_title = r'Time (micro secs)',
                yaxis_title = 'Displacement (mean subtracted)',
                legend      = dict(
                    orientation = "h",
                    yanchor     = "bottom",
                    y           = 1.02,
                    xanchor     = "right",
                    x           = 1
                )
            )

            fig.write_html("./graphs/" + csv[68:78] + ".html")

            # fig.show()
get_htmls(csv,face_csv)