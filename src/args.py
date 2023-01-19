
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input", 
                    default="/home/garima/Desktop/UoL_Projects/Gaze_Tracking/mediapipe_browser/converted.mp4", help="Input video path")

parser.add_argument("-o", "--output",
                    default="None", help="output video path")
parser.add_argument("-t", "--temp",
                    default="./temp", help="temporary folder path")

parser.add_argument("-of", "--optical_flow",
                    default="None", help="input video for optical flow")


parser.add_argument("-ce", "--csvs_eye",
                    default="None", help="folder containing iris and face csv")

