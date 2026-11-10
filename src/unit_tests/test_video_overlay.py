"""Run with PYTHONPATH=src python src/unit_tests/test_video_overlay.py."""
import numpy as np
from PIL import Image
from main.record_policy_videos import overlay

frame=Image.new('RGB',(200,100),'black')
positions=np.array([[-.8,-.8,0],[.6,-.8,0],[.9,-.8,0]])
matrices=np.repeat(np.eye(4)[None],3,axis=0)
rendered=overlay(frame,positions,matrices,matrices,0,1,'Test')
assert rendered.getpixel((160,80)) == (255,128,64)  # +1 drawn
assert rendered.getpixel((190,80)) == (0,0,0)       # +2 outside requested horizon
assert frame.getpixel((160,80)) == (0,0,0)         # source frame preserved
# At the end of the lap there are no future dots and no index overflow.
overlay(frame,positions,matrices,matrices,2,10,'Test')
print('Video overlay checks passed')
