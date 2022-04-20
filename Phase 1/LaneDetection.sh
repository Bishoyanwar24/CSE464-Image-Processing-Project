#!/bin/sh

#First Argument 
#Input Video Path
input="C:\Users\Bishoy Anwar\OneDrive\Desktop\challenge_video.mp4"

#Second Argument
#Output Video Path
output="shell exe/challenge_video"

#Third Argument
# 1 --> run debuging mode
# 0 --> run normal mode

ipython LaneDetection.py "$input" "$output" 1
