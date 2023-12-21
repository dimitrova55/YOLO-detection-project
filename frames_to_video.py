import cv2
import os
from PIL import Image
import numpy as np
import shutil



def resize_image(image_path, target_size):

    # Loop through image files, resize, and save to the output directory
    
    image = Image.open(image_path)
    resized_image = image.resize(target_size, Image.LANCZOS)
    
    # Get the original image file name
    file_name = os.path.basename(image_path)
    
    # Save the resized image to the output directory
    # output_path = os.path.join(output_directory, file_name)
    # resized_image.save(output_path)
    
    return resized_image
    
    
            
if __name__ == "__main__":

    # Define the output video file name and properties
    output_video_file = 'right_view.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    frame_rate = 30.0  # Frames per second
    frame_size = (3840, 2160)  # Frame size (width, height)
    new_frame_size = (640, 480) # (1280, 720)

    # Create a VideoWriter object to write the video
    # the frame size should be changed if we are downsampling the images
    out = cv2.VideoWriter(output_video_file, fourcc, frame_rate, new_frame_size)

    # Directory containing frames as image files
    input_directory = 'C:\\Users\\user\\Downloads\\yolo_data\\test_data\\right_sided_view'
    # output_directory = 'C:\\Users\\user\\Downloads\\abandoned-object-detection-opencv\\abandoned_11\\3093125_resized'
    # os.makedirs(output_directory, exist_ok=True)
    
    # List all folders
    folders = sorted(os.listdir(input_directory))
    
    for folder in folders:
        
        # List all image files in the directory
        # frame_files = sorted([f for f in os.listdir(frames_directory) if f.endswith('.jpg')])
        frame_files = sorted(os.listdir(os.path.join(input_directory, folder)))

        # Loop through the image files and add them to the videos
        for frame_file in frame_files:        
            frame_path = os.path.join(input_directory, folder, frame_file)
            print(frame_path)
            
            # frame = cv2.imread(frame_path)        
            # if frame is not None:
            #     out.write(frame)
            
            # if we want to resize (downsample) the images first and then save the video with a lower resolution: 
            # resizing the framesimage
            resized_frame = resize_image(frame_path, new_frame_size)        
                        
            # Convert the resized image to a numpy array
            frame_np = cv2.cvtColor(np.array(resized_frame), cv2.COLOR_RGB2BGR)
            
            # Write the frame to the video    
            if frame_np is not None:
                # Write the same frame 3 times to make up for the missing ones and preserve the fps to 30
                for _ in range(3):
                    out.write(frame_np)
            
        # end of internal loop
    # end of external loop
        
    # Release the VideoWriter object
    out.release()

    print(f"Video saved as {output_video_file}")
