import os
import random
import shutil

# Paths to image and annotation folders
image_folder = 'C:\\Users\\user\\Downloads\\CCTV 추적 영상\\Training\\wheelchair_resized'
annotation_folder = 'C:\\Users\\user\\Downloads\\CCTV 추적 영상\\Training\\wheelchair_label'

# Paths for training and validation sets
train_image_folder = 'C:\\Users\\user\\Downloads\\yolo_data\\dataset\\images\\train'
train_annotation_folder = 'C:\\Users\\user\\Downloads\\yolo_data\\dataset\\labels\\train'
val_image_folder = 'C:\\Users\\user\\Downloads\\yolo_data\\dataset\\images\\valid'
val_annotation_folder = 'C:\\Users\\user\\Downloads\\yolo_data\\dataset\\labels\\valid'

# Get lists of image and annotation files
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])
annotation_files = sorted([f for f in os.listdir(annotation_folder) if f.endswith('.txt')])

# Shuffle the lists while maintaining correspondence
combined = list(zip(image_files, annotation_files))
random.shuffle(combined)
image_files, annotation_files = zip(*combined)

# Calculate the split point for training and validation sets
split_index = int(len(image_files) * 0.9)

# Distribute files to training and validation sets
train_images = image_files[:split_index]
train_annotations = annotation_files[:split_index]
val_images = image_files[split_index:]
val_annotations = annotation_files[split_index:]

# Copy images and annotations to respective folders
for image, annotation in zip(train_images, train_annotations):
    shutil.copy(os.path.join(image_folder, image), os.path.join(train_image_folder, image))
    shutil.copy(os.path.join(annotation_folder, annotation), os.path.join(train_annotation_folder, annotation))

print("Train split Done!")

for image, annotation in zip(val_images, val_annotations):
    shutil.copy(os.path.join(image_folder, image), os.path.join(val_image_folder, image))
    shutil.copy(os.path.join(annotation_folder, annotation), os.path.join(val_annotation_folder, annotation))

print("Validation split Done!")