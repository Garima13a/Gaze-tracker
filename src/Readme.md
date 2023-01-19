# Gaze Tracking

This project involves 3 different experimentations:

## Video Smoothing & Stabilization

Video stabilization reduces the effect of camera motion on the final video. The motion of the camera would be a translation ( i.e. movement in the x, y, z-direction ) or rotation (yaw, pitch, roll). The technique used in this experimentation is '**Video Stabilization Using Point Feature Matching**'. This method involves tracking a few feature points between two consecutive frames. The tracked features allow us to estimate the motion between frames and compensate for it.

First thing to do in gaze tracking is to stabilize the raw input video. You can do this by running the command below.

`python main.py -i <path of input video> -o <output video path> -t <temporary folder>`

***Note**: Create a temporary folder (if not present) to store the output frames which you later be used to convert into a video*

## Optical Flow (Optional)
Optical flow is a task of per-pixel motion estimation between two consecutive frames in one video. Basically, the Optical Flow task implies the calculation of the shift vector for pixel as an object displacement difference between two neighboring images. The main idea of Optical Flow is to estimate the object’s displacement vector caused by it’s motion or camera movements.

You can do this by running the command below:

`python optical.py -of <path of video>`

## Running Mediapipe Iris
 
* Install mediapipe as directed in 
<a href="https://google.github.io/mediapipe/getting_started/install.html" target="_blank">Mediapipe installation</a>

* Put model/iris_tracking_cpu_video_input_copy file into this path:

 /mediapipe/bazel-bin/mediapipe/examples/desktop/iris_tracking 

* Open terminal:

`cd mediapipe`

* Then run below command:

`bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/iris_tracking:iris_tracking_cpu_video_input`

* Add path to your video that needs to be processed in mediapipe in 'path of input video'. Ex: /home/garima/Desktop/input.mp4

* Add path to your output video 'path of output video'

**Note**: You need a target video which would get replaced as the output video. I recommend making a copy of input video and name it 'output.mp4' and put that path as a 'path of output video'. Ex: /home/garima/Desktop/output.mp4

* Add path where you would want your csv files to be generated. 

**Note**: Mediapipe takes this path as 'path + name of the csv'. Ex: /home/garima/Desktop/outputfile. It will generate a csv named outputfile.csv in Desktop folder.

`bazel-bin/mediapipe/examples/desktop/iris_tracking/iris_tracking_cpu_video_input_copy --calculator_graph_config_file=mediapipe/graphs/iris_tracking/iris_tracking_cpu_video_input.pbtxt --input_side_packets=input_video_path=<path of input video>,output_video_path=<path of output video> --output_stream iris_landmarks --output_stream_file <path of output csv file> --logtostderr=1`

