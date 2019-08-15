import datetime
import os

import face_recognition
import pandas as pd

from configs import ALLOWED_EXTENSIONS


def timesince(time, percent_done):
    """Returns a string of the time taken so far, and an estimate of the time remaining (given the time taken so far and the percentage complete"""
    diff = datetime.datetime.now() - time
    days, seconds = diff.days, diff.seconds
    hours = days * 24 + seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    try:
        remaining_diff = (diff / percent_done) * 100 - diff
        remaining_days, remaining_seconds = remaining_diff.days, remaining_diff.seconds
        remaining_hours = remaining_days * 24 + remaining_seconds // 3600
        remaining_minutes = (remaining_seconds % 3600) // 60
        remaining_seconds = remaining_seconds % 60
        return (str(hours) + " hours, " + str(minutes) + " minutes, " + str(seconds) + " seconds taken so far" + "\n" +
                "Estimated " + str(remaining_hours) + " hours, " + str(remaining_minutes) + " minutes, " + str(
                    remaining_seconds) + " seconds to completion")
    except:
        return ("Cannot calculate times done and remaining at this time")


def time_diff(time1, time2):
    """Returns a sting of duration between two times"""
    diff = time2 - time1
    days, seconds = diff.days, diff.seconds
    hours = days * 24 + seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return str(hours) + " hours, " + str(minutes) + " minutes, " + str(seconds) + " seconds taken"


def allowed_file(filename):
    """A check if the file in question has an allowed extension type"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def encodings_builder(base_directory, image_no_max, image_attempt_no_max, attempting_all, encodings_start_time,
                      start_time):
    all_encodings = []
    person_no = 0
    image_no = 1
    image_attempt_no = 1
    failed_attempts = 0
    images_without_faces = []
    for person in os.listdir(base_directory):
        person_path = os.path.join(base_directory, person)
        person_no += 1
        if person != "Thumbs.db":
            this_persons_successful_encodings = 0
            for image in os.listdir(person_path):
                image_path = os.path.join(person_path, image)
                if image != "Thumbs.db" and allowed_file(image):
                    loaded_image = face_recognition.load_image_file(image_path)
                    if attempting_all:
                        print(
                            "Now scanning " + person + ", image " + str(image_attempt_no) + " of " + str(image_no_max) +
                            " (" + str(round(100 * (image_attempt_no - 1) / image_no_max, 2)) + "% completed)." +
                            " (" + str(failed_attempts) + " images have failed.)\n" +
                            timesince(start_time, 100 * (image_attempt_no - 1) / image_no_max) + " of scans")
                    else:
                        print("Now scanning " + person + ", image " + str(image_no) + " of " + str(image_no_max) +
                              " (" + str(round(100 * (image_no - 1) / image_no_max, 2)) + "% completed). " +
                              timesince(start_time, 100 * (image_no - 1) / image_no_max) + " of scans")
                    encodings = face_recognition.face_encodings(loaded_image)
                    if len(encodings) > 0:
                        all_encodings.append([person_path, image_path, encodings])
                        if (image_no >= image_no_max) or (image_attempt_no >= image_attempt_no_max):
                            return all_encodings, image_no, person_no, image_attempt_no, failed_attempts, images_without_faces
                        image_no += 1
                        image_attempt_no += 1
                        this_persons_successful_encodings += 1
                        print("")
                    else:
                        print("No face found this image")
                        print("")
                        image_attempt_no += 1
                        failed_attempts += 1
                        images_without_faces.append(image_path)
            if this_persons_successful_encodings == 0:
                person_no -= 1
    return all_encodings, image_no, person_no, image_attempt_no, failed_attempts, images_without_faces

def get_number_faces_to_scan(base_directory):
    print("")
    image_no_max = input("How many images do you want to compare? (leave blank for all)")

    # Count the number of folders (people) in dataset, if no number defined then compare with all

    person_count = len(([len(files) for r, d, files in os.walk(base_directory)])) - 1

    # Count images

    if image_no_max == "":
        attempting_all = True
    else:
        image_no_max = int(image_no_max)
        attempting_all = False

    if attempting_all:
        print("")
        print("Counting the images...\n")
        file_count = 0
        person_no = 0
        for _, dirs, files in os.walk(base_directory):
            person_no += 1
            file_count += len([x for x in files if x != "Thumbs.db"])
        image_no_max = file_count

    print(str(image_no_max) + " files to attempt to scan and compare")
    print("")
    return image_no_max, attempting_all, person_count

def encodings_comparer(all_encodings):

    all_comparisons = []
    image_counter = 1
    same_face_distances = []
    different_face_distances = []
    comparison_counter = 0
    total_comparisons = (len(all_encodings) * (len(all_encodings) - 1)) / 2
    start_time = datetime.datetime.now()

    # Looping over combinations of images, and getting face_distance for each pair

    for image in range(1, len(all_encodings)):
        for image2 in range(0, image):
            distances = []
            for face1 in range(len(all_encodings[image][2])):
                for face2 in range(len(all_encodings[image2][2])):
                    distances.append(face_recognition.face_distance([all_encodings[image][2][face1]],
                                                                    all_encodings[image2][2][face2])[0])
            if len(distances) > 0:
                comparison_counter += 1
                if comparison_counter % 1000000 == 0:
                    print(
                            "Comparison number " + str(comparison_counter) + " of " + str(
                        round(total_comparisons)) + " (" +
                            str(round(100 * comparison_counter / total_comparisons, 2)) + "% completed). " +
                            timesince(start_time,
                                      100 * (comparison_counter / total_comparisons)) + " of comparisons")
                    print("")
                distance = min(distances)
                same = all_encodings[image][0] == all_encodings[image2][0]
                if same:
                    same_face_distances.append((all_encodings[image][1], all_encodings[image2][1], distance))
                else:
                    different_face_distances.append((all_encodings[image][1], all_encodings[image2][1], distance))
        image_counter += 1

    same_face_distances_df = pd.DataFrame(same_face_distances, columns=['path1', 'path2', 'distance'])
    different_face_distances_df = pd.DataFrame(different_face_distances, columns=['path1', 'path2', 'distance'])

    same_face_distances = list(same_face_distances_df['distance'])
    different_face_distances = list(different_face_distances_df['distance'])

    print(datetime.datetime.now().strftime("%Y_%m_%d %H:%M:%S") + " Image comparisons completed!")
    print("")

    return same_face_distances, different_face_distances, comparison_counter, different_face_distances_df, same_face_distances_df