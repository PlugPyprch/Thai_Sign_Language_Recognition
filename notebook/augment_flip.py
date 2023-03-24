import moviepy.editor as mpy
import os

# Path to the folder containing the input videos
input_folder = "videos/unwell/"

# List of input videos
videos = []

# Iterate over all files in the input folder
for file in os.listdir(input_folder):
    # Check if the file is a video file
    if file.endswith(".mp4"):
        # Add the file to the list of videos
        # print(file[2:-4])
        videos.append(os.path.join(input_folder, file))
        

for video in videos:
    # print(video[16:])
    
    # Load the video clip
    clip = mpy.VideoFileClip(video)

    # Apply the horizontal flip transformation to the clip
    transformed_clip = clip.fx(mpy.vfx.mirror_x)

    # Construct the output filename
    output_filename = "6_" + str(int(video[16:-4])+100) + ".mp4"

    # Save the transformed clip as a new mp4 file
    transformed_clip.write_videofile('videos/unwell/' + output_filename)

