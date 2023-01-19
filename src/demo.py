from pathlib import Path
import os
ip = '/home/garima/Desktop/hello.mp4'

parent_dir = os.path.split(ip)[0]
print(parent_dir)

Path(parent_dir + "/optical_flow_temp").mkdir(parents=True, exist_ok=True)
