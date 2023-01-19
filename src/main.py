from stabilize import *
from frameToVid import *
from args import parser
args    = parser.parse_args()

input_video_path = args.input
output_video_path = args.output
temp_folder = args.temp

get_stable_video(input_video_path,output_video_path,temp_folder)

frame_to_vid(temp_folder,temp_folder)
print('The output video is in' + temp_folder)