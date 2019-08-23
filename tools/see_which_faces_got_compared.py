"""
CHECKING THAT THE RIGHT FACES IN IMAGES WERE USED FOR THE COMPARISONS
"""

import os

import cv2
import face_recognition
import numpy as np
import pandas as pd

from configs import results_directory

# Which file to look at

which_results = 'different_faces_looking_similar'
#which_results = 'same_face_looking_different'

# Get the most recent file of that type, extract the paths

all_results_files = os.listdir(results_directory)
most_recent_results_file_of_specified_type = [i for i in all_results_files if (which_results in i)][-1]
full_path = os.path.join(results_directory, most_recent_results_file_of_specified_type)
paths_df = pd.read_csv(full_path)[['path1', 'path2']]
paths_list = paths_df.values.tolist()

# Loop over those paths

for paths in paths_list:

    print('\nImage pair: ')
    print(paths)
    images = []
    encodings = []

    for path in paths:

        # Get the image, face locations, and encodings for this image

        image = face_recognition.load_image_file(path)
        img = cv2.imread(path)
        face_locations = face_recognition.face_locations(image)
        this_image_encodings = face_recognition.face_encodings(image)
        encodings.append(this_image_encodings)

        # Add rectangles around faces (red for first face, blue for second face)
        col_i = 0
        for (top, right, bottom, left) in face_locations:
            cols = [(0, 0, 255), (255, 0, 0)]
            img = cv2.rectangle(img, (left, top), (right, bottom), cols[col_i], 2)
            col_i += 1
            img = cv2.resize(img, (250, 250))

        images.append(img)

    # Print the face distances, in order
    for face in encodings[1]:
        print('Face distance: ' + str(round(face_recognition.face_distance(encodings[0], face)[0], 3)))

    images_combined = np.concatenate(images, axis=1)

    # Display the resulting image
    cv2.imshow('image', images_combined)
    cv2.waitKey(0)
