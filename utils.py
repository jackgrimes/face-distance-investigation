import datetime
import os

import cv2
import face_recognition
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.misc import imsave
from scipy.stats import norm
from sklearn import metrics
from wordcloud import WordCloud

from configs import ALLOWED_EXTENSIONS, IMAGES_TO_EXCLUDE, N_LOOKALIKES_AND_DIFFERENT_LOOKING_SAME_PEOPLE_TO_INCLUDE, \
    ACTUALLY_SAME_PEOPLE
from configs import lfw_path, results_directory


def timesince(time, percent_done):
    """
    Returns a string of the time taken so far, and an estimate of the time remaining (given the time taken so far and the percentage complete
    :param time:
    :param percent_done:
    :return:
    """
    now = datetime.datetime.now()
    diff = now - time
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
        print(str(hours) + " hours, " + str(minutes) + " minutes, " + str(seconds) + " seconds taken so far")
        print("Estimated " + str(remaining_hours) + " hours, " + str(remaining_minutes) + " minutes, " + str(
            remaining_seconds) + " seconds to completion")
        print("Estimated completion time: " + (time + ((now - time) / percent_done) * 100).strftime("%H:%M:%S"))
    except:
        print("Cannot calculate times done and remaining at this time")


def time_diff(time1, time2):
    """
    Returns a sting of duration between two times
    :param time1:
    :param time2:
    :return:
    """
    diff = time2 - time1
    days, seconds = diff.days, diff.seconds
    hours = days * 24 + seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return str(hours) + " hours, " + str(minutes) + " minutes, " + str(seconds) + " seconds taken"


def allowed_file(filename):
    """
    A check if the file in question has an allowed extension type
    :param filename:
    :return:
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def print_updates_get_encodings(person, number_of_people_to_scan, encodings_start_time, lists_of_images):
    """
    So that the user knows roughly how long it will take to finish getting the face encodings
    :param person:
    :param n_people:
    :param images_to_attempt:
    :param encodings_start_time:
    :param lists_of_images:
    :return:
    """
    print("Getting face encodings for images of " + os.path.basename(person))
    print("Person number " + str(len(lists_of_images['people_attempted']) + 1) + " of " + str(
        number_of_people_to_scan) + ", " + str(
        round(100 * (len(lists_of_images['people_attempted'])) / number_of_people_to_scan)) + "% complete")
    timesince(encodings_start_time, 100 * (len(lists_of_images['people_attempted'])) / number_of_people_to_scan)


def find_which_people_and_images_to_scan(attempting_all, lfw_path, image_no_max):
    """

    :param attempting_all:
    :param lfw_path:
    :param image_no_max:
    :return:
    """
    people = [folder for folder in os.listdir(lfw_path) if ("." not in folder)]
    people_directory_full_paths = [os.path.join(lfw_path, person) for person in os.listdir(lfw_path) if
                                   (".db" not in person)]
    people_image_nos = [len([file for file in os.listdir(person_directory) if
                             ((".db" not in file) and (os.path.join(person_directory, file) not in IMAGES_TO_EXCLUDE))])
                        for
                        person_directory in people_directory_full_paths]
    people_image_nos_cum = list(np.cumsum(people_image_nos))

    if not attempting_all:
        first_exceeding = [(x > image_no_max) for x in people_image_nos_cum].index(True)
        people = people[:first_exceeding + 1]
        images_to_attempt = image_no_max
    else:
        images_to_attempt = people_image_nos_cum[-1]

    n_people = len(people)

    return people, images_to_attempt, n_people


def get_this_persons_encodings(person_path, IMAGES_TO_EXCLUDE):
    """

    :param person_path:
    :return:
    """
    images_of_this_person = [os.path.join(person_path, file) for file in os.listdir(person_path) if
                             (".db" not in file)]

    images_this_person_in_IMAGES_TO_EXCLUDE = [image for image in images_of_this_person if (image in IMAGES_TO_EXCLUDE)]
    images_this_person_not_in_IMAGES_TO_EXCLUDE = [image for image in images_of_this_person if
                                                   (image not in IMAGES_TO_EXCLUDE)]

    face_rec_loaded_images_of_this_person = [[image, face_recognition.load_image_file(image)] for image in
                                             images_this_person_not_in_IMAGES_TO_EXCLUDE]
    encodings_images_this_person_not_in_IMAGES_TO_EXCLUDE = [[image[0], face_recognition.face_encodings(image[1])] for
                                                             image in
                                                             face_rec_loaded_images_of_this_person]

    return encodings_images_this_person_not_in_IMAGES_TO_EXCLUDE, images_this_person_not_in_IMAGES_TO_EXCLUDE, images_this_person_in_IMAGES_TO_EXCLUDE


def check_for_no_or_multiple_images_in_photo(encodings_all_images_this_person, images_of_this_person):
    """

    :param encodings_all_images_this_person:
    :param images_of_this_person:
    :return:
    """
    number_faces_found_in_each_image = [len(encodings_list[1]) for encodings_list in
                                        encodings_all_images_this_person]
    images_without_faces = [(number == 0) for number in number_faces_found_in_each_image]

    image_paths_without_faces = [images_of_this_person[i] for i in range(len(images_of_this_person)) if
                                 images_without_faces[i]]

    encodings_for_images_with_faces = [encodings_all_images_this_person[i][1] for i in
                                       range(len(images_of_this_person)) if
                                       (not images_without_faces[i])]
    paths_for_images_with_faces = [images_of_this_person[i] for i in range(len(images_of_this_person)) if
                                   (not images_without_faces[i])]

    return image_paths_without_faces, number_faces_found_in_each_image, encodings_for_images_with_faces, paths_for_images_with_faces


def select_right_face_encodings_from_each_image(paths_for_images_with_faces, encodings_for_images_with_faces):
    """

    :param paths_for_images_with_faces:
    :param selected_encodings_from_this_image:
    :return:
    """
    images_with_unidentifiable_faces = []
    selected_encodings_from_this_persons_images = []
    for i in range(len(paths_for_images_with_faces)):
        this_image_encodings = encodings_for_images_with_faces[i]
        if len(this_image_encodings) == 1:
            selected_encodings_from_this_image = this_image_encodings
            selected_encodings_from_this_persons_images.append(selected_encodings_from_this_image)
        else:
            encodings_other_images = [x for x in encodings_for_images_with_faces[:i] + encodings_for_images_with_faces[
                                                                                       i + 1:]]
            flattened_encodings_other_images = [item for sublist in encodings_other_images for item in sublist]
            if (len(flattened_encodings_other_images) > 0) and (len(paths_for_images_with_faces) > 2):
                this_image_face_distances_to_other_image_faces = []
                for j, this_encoding_set in enumerate(encodings_for_images_with_faces[i]):
                    distances = face_recognition.face_distance(flattened_encodings_other_images,
                                                               this_encoding_set)
                    this_image_face_distances_to_other_image_faces.append(
                        [j, sum(distances.T.tolist()) / len(distances.T.tolist())])
                this_image_face_distances_to_other_image_faces_df = pd.DataFrame(
                    this_image_face_distances_to_other_image_faces)
                this_image_face_distances_to_other_image_faces_df = this_image_face_distances_to_other_image_faces_df.rename(
                    columns={0: 'face', 1: 'distance'})
                face_most_similar_to_faces_in_other_images_index = this_image_face_distances_to_other_image_faces_df[
                    this_image_face_distances_to_other_image_faces_df['distance'] ==
                    this_image_face_distances_to_other_image_faces_df['distance'].min()][
                    'face'].values[0]
                selected_encodings_from_this_image = [this_image_encodings[
                                                          face_most_similar_to_faces_in_other_images_index]]
                selected_encodings_from_this_persons_images.append(selected_encodings_from_this_image)
            else:
                print("Cannot tell which face is this persons!")
                images_with_unidentifiable_faces.append(paths_for_images_with_faces[i])

    return selected_encodings_from_this_persons_images, images_with_unidentifiable_faces, paths_for_images_with_faces


def put_selected_encodings_into_df(selected_encodings_from_this_image, paths_for_images_with_faces, person_path):
    n_encodings_this_person = sum([len(x) for x in selected_encodings_from_this_image])

    if n_encodings_this_person > 0:
        this_persons_encodings = pd.DataFrame(
            list(zip(paths_for_images_with_faces, selected_encodings_from_this_image)))
        this_persons_encodings.columns = ['image_path', 'encodings']
        this_persons_encodings['person_path'] = os.path.dirname(paths_for_images_with_faces[0])
        this_persons_first_name = os.path.basename(person_path).split('_')[0]
    else:
        this_persons_encodings = pd.DataFrame()
        this_persons_first_name = ''

    return this_persons_encodings, this_persons_first_name


def encodings_builder(lfw_path,
                      number_of_people_to_scan,
                      peoples_faces_to_scan,
                      IMAGES_TO_EXCLUDE):
    """

    :param lfw_path:
    :param image_no_max:
    :param attempting_all:
    :return:
    """
    encodings_start_time = datetime.datetime.now()
    print("\n" + encodings_start_time.strftime("%Y_%m_%d__%H:%M:%S") + " Doing the encodings..." + "\n")

    all_encodings = pd.DataFrame()

    lists_of_images = {'images_attempted': [],
                       'people_attempted': [],
                       'people_with_encodings': [],
                       'images_with_encodings': [],
                       'IMAGES_TO_EXCLUDE_excluded': [],
                       'photos_with_no_faces_found_paths': [],
                       'photos_with_multiple_faces_and_no_other_images_to_compare_with': [],
                       'people_names_corrected_to_other_people': [],
                       'images_for_people_names_corrected_to_other_people': [],
                       'all_peoples_first_names': [],
                       }

    for person_number, person in enumerate(peoples_faces_to_scan):

        person_path = os.path.join(lfw_path, person)

        print_updates_get_encodings(person, number_of_people_to_scan, encodings_start_time, lists_of_images)

        (encodings_images_this_person_not_in_IMAGES_TO_EXCLUDE,
         images_this_person_not_in_IMAGES_TO_EXCLUDE,
         images_this_person_in_IMAGES_TO_EXCLUDE) = get_this_persons_encodings(person_path,
                                                                               IMAGES_TO_EXCLUDE)

        if len(images_this_person_not_in_IMAGES_TO_EXCLUDE) > 0:
            lists_of_images['people_attempted'].append(person)

        lists_of_images['images_attempted'].extend(encodings_images_this_person_not_in_IMAGES_TO_EXCLUDE)
        lists_of_images['IMAGES_TO_EXCLUDE_excluded'].extend(images_this_person_in_IMAGES_TO_EXCLUDE)

        (image_paths_without_faces,
         number_faces_found_in_each_image,
         encodings_for_images_with_faces,
         paths_for_images_with_faces) = check_for_no_or_multiple_images_in_photo(
            encodings_images_this_person_not_in_IMAGES_TO_EXCLUDE,
            images_this_person_not_in_IMAGES_TO_EXCLUDE)

        lists_of_images['photos_with_no_faces_found_paths'].extend(image_paths_without_faces)

        (selected_encodings_from_this_persons_images,
         images_with_unidentifiable_faces,
         paths_for_images_with_faces) = select_right_face_encodings_from_each_image(
            paths_for_images_with_faces, encodings_for_images_with_faces)

        lists_of_images['photos_with_multiple_faces_and_no_other_images_to_compare_with'].extend(
            images_with_unidentifiable_faces)

        (this_persons_encodings,
         this_persons_first_name) = put_selected_encodings_into_df(selected_encodings_from_this_persons_images,
                                                                   paths_for_images_with_faces, person_path)

        if this_persons_encodings.shape[0] > 0:
            lists_of_images['images_with_encodings'].extend(this_persons_encodings['image_path'].tolist())

        # Correct person_path, where appropriate, if this person is also found elsewhere in lfw under a different name
        if os.path.basename(person_path) in ACTUALLY_SAME_PEOPLE.keys():
            lists_of_images['people_names_corrected_to_other_people'].append(os.path.basename(person_path))
            lists_of_images['images_for_people_names_corrected_to_other_people'].extend(
                this_persons_encodings['image_path'])
            person = ACTUALLY_SAME_PEOPLE[os.path.basename(person_path)]
            person_path = os.path.join(os.path.dirname(person_path), person)
            this_persons_encodings['person_path'] = person_path
        elif this_persons_encodings.shape[0] > 0:
            lists_of_images['people_with_encodings'].append(person)
            lists_of_images['all_peoples_first_names'].append(this_persons_first_name)

        all_encodings = pd.concat([all_encodings, this_persons_encodings])

        print("")

        lists_of_images['all_peoples_first_names'] = [name for name in lists_of_images['all_peoples_first_names'] if
                                                      (name != '')]

    all_encodings = all_encodings.reset_index(drop=True)
    all_encodings = all_encodings[['person_path', 'image_path', 'encodings']]

    return all_encodings, encodings_start_time, lists_of_images


def get_number_faces_to_scan(lfw_path, overall_start_time):
    """

    :param lfw_path:
    :param overall_start_time:
    :return:
    """

    number_of_people_to_scan = input("\nHow many people's images do you want to compare? (leave blank for all)")

    people_in_lfw_path = [os.path.join(lfw_path, person) for person in os.listdir(lfw_path)]

    # Count images

    if number_of_people_to_scan == "":
        attempting_all = True
        peoples_faces_to_scan = people_in_lfw_path
        number_of_people_to_scan = len(peoples_faces_to_scan)
    else:
        number_of_people_to_scan = int(number_of_people_to_scan)
        attempting_all = False
        peoples_faces_to_scan = people_in_lfw_path[0:number_of_people_to_scan]

    file_str_prefix = os.path.join(results_directory,
                                   overall_start_time.strftime("%Y_%m_%d %H_%M_%S_") + (
                                       "_attempting_all_images_" if attempting_all else ("_attempting_" + str(
                                           number_of_people_to_scan) + "_peoples_images_")))

    return number_of_people_to_scan, attempting_all, file_str_prefix, peoples_faces_to_scan


def perhaps_print_comparison_counter(comparison_counter, total_comparisons, comparisons_start_time):
    if comparison_counter % 1000000 == 0:
        print(
            "Comparison number " + str(comparison_counter) + " of " + str(
                round(total_comparisons)) + " (" +
            str(round(100 * comparison_counter / total_comparisons, 2)) + "% completed). ")
        timesince(comparisons_start_time, 100 * (comparison_counter / total_comparisons))
        print("")


def encodings_comparer(all_encodings):
    """

    :param all_encodings:
    :return:
    """
    comparisons_start_time = datetime.datetime.now()
    print(comparisons_start_time.strftime("%Y_%m_%d__%H:%M:%S") + " Doing the comparisons..." + "\n")

    same_face_distances = []
    different_face_distances = []
    comparison_counter = 0
    total_comparisons = (len(all_encodings) * (len(all_encodings) - 1)) / 2

    all_encodings_lol = all_encodings.values.tolist()

    # Looping over combinations of images, and getting face_distance for each pair

    for image in range(1, len(all_encodings_lol)):
        for image2 in range(0, image):

            distance = face_recognition.face_distance(all_encodings_lol[image][2], all_encodings_lol[image2][2][0])[0]

            comparison_counter += 1
            perhaps_print_comparison_counter(comparison_counter, total_comparisons, comparisons_start_time)

            same = all_encodings_lol[image][0] == all_encodings_lol[image2][0]

            if same:
                same_face_distances.append((all_encodings_lol[image][1], all_encodings_lol[image2][1], distance))
            else:
                different_face_distances.append((all_encodings_lol[image][1], all_encodings_lol[image2][1], distance))

    print(datetime.datetime.now().strftime("%Y_%m_%d__%H:%M:%S") + " Image comparisons completed!")

    same_face_distances_df = pd.DataFrame(same_face_distances, columns=['path1', 'path2', 'distance'])
    different_face_distances_df = pd.DataFrame(different_face_distances, columns=['path1', 'path2', 'distance'])

    return same_face_distances_df, different_face_distances_df, comparisons_start_time, comparison_counter


def plotter(fig_names, cumulative, same_face_distances_df, different_face_distances_df, comparison_counter,
            file_str_prefix, lists_of_images):
    """

    :param fig_names:
    :param cumulative:
    :param same_face_distances_df:
    :param different_face_distances_df:
    :param comparison_counter:
    :param image_no:
    :param person_no:
    :param file_str_prefix:
    :return:
    """

    same_face_distances = same_face_distances_df['distance']
    different_face_distances = different_face_distances_df['distance']

    fig = plt.figure()
    fig.clf()
    xmin, xmax = min(min(same_face_distances), min(different_face_distances)), max(max(same_face_distances),
                                                                                   max(different_face_distances))
    bins = np.linspace(xmin, xmax, 100)
    ax = fig.add_subplot(111)
    plt.hist(same_face_distances, bins, alpha=0.5, label='same faces', density=True, edgecolor='black',
             linewidth=1.2, color='green', cumulative=cumulative)
    plt.hist(different_face_distances, bins, alpha=0.5, label='different faces', density=True, edgecolor='black',
             linewidth=1.2, color='blue', cumulative=cumulative)
    plt.legend(loc='upper right')
    plt.text(0.15, 0.9, str(comparison_counter) + " comparisons of \n" + str(
        len(lists_of_images['images_with_encodings'])) + " images of \n " + str(
        len(lists_of_images['people_with_encodings'])) + " people",
             ha='center', va='center', transform=ax.transAxes)
    fig.set_size_inches(12, 8)
    plt.xlabel("Face distance")
    plt.ylabel("Probability density")
    fig.savefig(file_str_prefix + fig_names[0] + '.png')

    # Add fitted distributions

    points_for_lines = np.linspace(xmin, xmax, 10000)
    mu_sf, std_sf = norm.fit(same_face_distances)
    mu_df, std_df = norm.fit(different_face_distances)

    if cumulative != False:
        p_sf = norm.cdf(points_for_lines, mu_sf, std_sf)
        p_df = norm.cdf(points_for_lines, mu_df, std_df)
        title = "Fit results: mu = {}, std = {}, mu = {}, std = {}".format(round(mu_sf, 2), round(std_sf, 2),
                                                                           round(mu_df, 2), round(std_df, 2))

    else:
        p_sf = norm.pdf(points_for_lines, mu_sf, std_sf)
        p_df = norm.pdf(points_for_lines, mu_df, std_df)
        title = "Fit results: mu = {}, std = {}, mu = {}, std = {}".format(round(mu_sf, 2), round(std_sf, 2),
                                                                           round(mu_df, 2), round(std_df, 2))

    plt.plot(points_for_lines, p_sf, 'k', linewidth=2)
    plt.plot(points_for_lines, p_df, 'k', linewidth=2)
    plt.title(title)
    fig.savefig(file_str_prefix + fig_names[1] + '.png')

    return ax


def roc_auc(same_face_distances_df, different_face_distances_df, ax, comparison_counter, lists_of_images,
            file_str_prefix):
    """

    :param same_face_distances_df:
    :param different_face_distances_df:
    :param ax:
    :param comparison_counter:
    :param lists_of_images:
    :param file_str_prefix:
    :return:
    """

    same_face_distances = same_face_distances_df['distance']
    different_face_distances = different_face_distances_df['distance']

    scores = np.concatenate((same_face_distances, different_face_distances), axis=0)
    y = np.concatenate((np.array([1] * len(same_face_distances)), np.array([0] * len(different_face_distances))),
                       axis=0)
    fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
    auc = metrics.auc(tpr, fpr)
    fig = plt.figure()
    fig.set_size_inches(12, 8)
    plt.plot(tpr, fpr)
    plt.xlabel("True positive rate")
    plt.ylabel("False positive rate")
    title = "ROC Curve (AUC = {})".format(round(auc, 4))
    plt.title(title)
    plt.text(0.9, 0.1,
             str(comparison_counter) + " comparisons of \n" + str(
                 len(lists_of_images['images_with_encodings'])) + " images of \n " + str(
                 len(lists_of_images['people_with_encodings'])) + " people",
             ha='center', va='center', transform=ax.transAxes)
    fig.savefig(file_str_prefix + r'_3_ROC.png')
    print(datetime.datetime.now().strftime("%Y_%m_%d__%H:%M:%S") + " Graphs completed.")
    print("")


def do_precision_recall(same_face_distances_df, different_face_distances_df, file_str_prefix, bin_boundaries,
                        output_filename):
    """

    :param same_face_distances_df:
    :param different_face_distances_df:
    :param file_str_prefix:
    :param bin_boundaries:
    :return:
    """
    same_face_distances = same_face_distances_df['distance']
    different_face_distances = different_face_distances_df['distance']

    same_faces_binned = pd.DataFrame({'same_faces': same_face_distances})
    same_faces_binned['same_faces_freq'] = pd.cut(same_faces_binned.same_faces, bin_boundaries)

    different_faces_binned = pd.DataFrame({'different_faces': different_face_distances})
    different_faces_binned['different_faces_freq'] = pd.cut(different_faces_binned.different_faces, bin_boundaries)

    precision_recall_table = pd.concat([same_faces_binned.same_faces_freq.value_counts(),
                                        different_faces_binned.different_faces_freq.value_counts()], axis=1,
                                       sort=True)

    precision_recall_table['same_faces_freq_cum'] = precision_recall_table['same_faces_freq'].cumsum() - \
                                                    precision_recall_table['same_faces_freq']

    precision_recall_table['different_faces_freq_cum'] = precision_recall_table['different_faces_freq'].cumsum() - \
                                                         precision_recall_table['different_faces_freq']
    precision_recall_table['different_faces_freq_anti_cum'] = sum(precision_recall_table['different_faces_freq']) - \
                                                              precision_recall_table['different_faces_freq_cum']

    precision_recall_table['cutoff'] = [
        float(str(x).split('(')[1].split(',')[0].replace(' ', '').replace("'", "").replace(']', '')) for x in
        list(precision_recall_table.index)]

    precision_recall_table['precision'] = precision_recall_table['same_faces_freq_cum'] / (
            precision_recall_table['same_faces_freq_cum'] + precision_recall_table['different_faces_freq_cum'])
    precision_recall_table['recall'] = precision_recall_table['same_faces_freq_cum'] / (
        sum(precision_recall_table['same_faces_freq']))

    precision_recall_table.to_csv(file_str_prefix + output_filename)


def precision_recall(same_face_distances_df, different_face_distances_df, file_str_prefix, doing_precision_recall):
    """

    :param same_face_distances_df:
    :param different_face_distances_df:
    :param file_str_prefix:
    :param doing_precision_recall:
    :return:
    """
    precision_recall_start_time = datetime.datetime.now()

    if doing_precision_recall:

        print(
            datetime.datetime.now().strftime("%Y_%m_%d__%H:%M:%S") + " Now doing precision and recall calculations...")
        print("")

        do_precision_recall(same_face_distances_df=same_face_distances_df,
                            different_face_distances_df=different_face_distances_df,
                            file_str_prefix=file_str_prefix,
                            bin_boundaries=[0, 0.1] + [round(x, 2) for x in np.arange(0.2, 1.1, 0.01)] + [1.2, 1.3, 1.4,
                                                                                                          1.5, 2],
                            output_filename='_4_precision_recall_table.csv'
                            )

        do_precision_recall(same_face_distances_df=same_face_distances_df,
                            different_face_distances_df=different_face_distances_df,
                            file_str_prefix=file_str_prefix,
                            bin_boundaries=[0, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.9, 1, 1.5, 2],
                            output_filename='_5_precision_recall_table_the_key_cutoffs.csv'
                            )

        print(datetime.datetime.now().strftime("%Y_%m_%d__%H:%M:%S") + " Precision and recall calculations completed.")

        return precision_recall_start_time


def select_top_unique_combos_and_output_to_csv(df, output_str, ascending, file_str_prefix):
    """

    :param df:
    :param output_str:
    :param ascending:
    :return:
    """

    df_sorted = df.sort_values(by=['distance'], ascending=ascending).head(1000)

    df_sorted['person1'] = df_sorted['path1'].apply(os.path.dirname)
    df_sorted['person2'] = df_sorted['path2'].apply(os.path.dirname)

    df_sorted['people_set'] = df_sorted[['person1', 'person2']].apply(lambda row: str(list(set(row.tolist()))),
                                                                      axis=1)

    df_sorted.drop_duplicates(subset=['people_set'], keep='first', inplace=True)
    df_sorted.reset_index(drop=True, inplace=True)
    df_sorted = df_sorted.head(N_LOOKALIKES_AND_DIFFERENT_LOOKING_SAME_PEOPLE_TO_INCLUDE)

    df_sorted['index'] = pd.Series(range(1, N_LOOKALIKES_AND_DIFFERENT_LOOKING_SAME_PEOPLE_TO_INCLUDE + 1))
    df_sorted = df_sorted[['index', 'path1', 'path2', 'distance']]

    df_sorted.to_csv(file_str_prefix + output_str + '.csv',
                     index=False)

    return df_sorted


def output_most_similar_different_people_and_most_different_same_faces(different_face_distances_df,
                                                                       same_face_distances_df, file_str_prefix):
    """

    :param different_face_distances_df:
    :param same_face_distances_df:
    :param file_str_prefix:
    :return:
    """

    # Most similar lookalikes
    different_face_distances_df_sorted = select_top_unique_combos_and_output_to_csv(different_face_distances_df,
                                                                                    '_6_lookalikes',
                                                                                    True, file_str_prefix)

    # Most different photos of the same person
    same_face_distances_df_sorted = select_top_unique_combos_and_output_to_csv(same_face_distances_df,
                                                                               '_7_different_looking_same_people',
                                                                               False,
                                                                               file_str_prefix)

    return different_face_distances_df_sorted, same_face_distances_df_sorted


def run_outputs(attempting_all, overall_start_time,
                encodings_start_time, comparisons_start_time, graph_start_time, precision_recall_start_time,
                file_str_prefix, lists_of_images):
    """

    :param attempting_all:
    :param overall_start_time:
    :param encodings_start_time:
    :param comparisons_start_time:
    :param graph_start_time:
    :param precision_recall_start_time:
    :param file_str_prefix:
    :param lists_of_images:
    :return:
    """

    completion_time = datetime.datetime.now()

    outputs_str = ""

    outputs_str += ("attempting_all was " + str(attempting_all) + "\n")

    n_people_lfw, n_images_lfw = count_images_lfw(lfw_path)

    outputs_str += ("\nLFW has " + str(n_images_lfw) + " images of " + str(n_people_lfw) + " people.\n")

    outputs_str += ("\nNumber of images excluded because the wrong face gets picked, etc: " + str(
        len(lists_of_images['IMAGES_TO_EXCLUDE_excluded'])) + "\n")
    if len(lists_of_images['IMAGES_TO_EXCLUDE_excluded']) > 0:
        outputs_str += ("\nImages excluded because the wrong face gets picked, etc:\n" +
                        "\n".join(lists_of_images['IMAGES_TO_EXCLUDE_excluded']) +
                        "\n\n")

    outputs_str += (str(len(lists_of_images['images_attempted'])) + " photos of " + str(
        len(lists_of_images['people_attempted'])) + " people remained to attempt.\n")

    outputs_str += ("Of these:")

    outputs_str += ("\n\t" + str(len(lists_of_images['images_with_encodings'])) + " photos of " + str(
        len(lists_of_images['people_with_encodings'])) + " people compared.")

    outputs_str += (
            "\n\t" + "Faces not found in " + str(len(lists_of_images['photos_with_no_faces_found_paths'])) + " images.")

    outputs_str += ("\n\t" + "Not sure which face to pick in " + str(
        len(lists_of_images['photos_with_multiple_faces_and_no_other_images_to_compare_with'])) + " images.")

    outputs_str += (
            "\n\t" + str(len(lists_of_images['images_for_people_names_corrected_to_other_people'])) + " photos of " + str(
        len(lists_of_images['people_names_corrected_to_other_people'])) + " people combined into other peoples' folders.\n\n")

    if len(lists_of_images['photos_with_no_faces_found_paths']) > 0:
        outputs_str += ("Images without faces:\n" +
                        "\n".join(lists_of_images['photos_with_no_faces_found_paths']) +
                        "\n\n")

    if len(lists_of_images['photos_with_multiple_faces_and_no_other_images_to_compare_with']) > 0:
        outputs_str += ("Images where not sure which face to pick:\n" +
                        "\n".join(lists_of_images['photos_with_multiple_faces_and_no_other_images_to_compare_with']) +
                        "\n\n")

    if len(lists_of_images['images_for_people_names_corrected_to_other_people']) > 0:
        outputs_str += ("Images reassigned into other people's folders:\n" +
                        "\n".join(lists_of_images['images_for_people_names_corrected_to_other_people']) +
                        "\n\n")

    outputs_str += ("Summary of time taken:\n\n" +
                    time_diff(overall_start_time,
                              encodings_start_time) + ' on initial prep, counting pictures etc\n' +
                    time_diff(encodings_start_time, comparisons_start_time) + ' on getting encodings\n' +
                    time_diff(comparisons_start_time, graph_start_time) + ' on comparing encodings\n' +
                    time_diff(graph_start_time, precision_recall_start_time) + ' on making graphs\n' +
                    time_diff(precision_recall_start_time,
                              completion_time) + ' on calculation precision_recall_table\n\n' +
                    time_diff(overall_start_time, completion_time) + ' in total')

    with open(file_str_prefix + "_11_run_notes.txt", "w") as file:
        file.write(outputs_str)

    print('\n' + outputs_str)

    print('\nAll Done!')


def all_graphs(same_face_distances_df, different_face_distances_df, comparison_counter, lists_of_images,
               file_str_prefix, doing_graphs, CUMULATIVE_GRAPHS):
    """

    :param same_face_distances_df:
    :param different_face_distances_df:
    :param comparison_counter:
    :param lists_of_images:
    :param file_str_prefix:
    :param doing_graphs:
    :return:
    """
    graph_start_time = datetime.datetime.now()
    print("\n" + graph_start_time.strftime("%Y_%m_%d__%H:%M:%S") + " Doing the graphs..." + "\n")

    if doing_graphs:
        print(datetime.datetime.now().strftime("%Y_%m_%d__%H:%M:%S") + " Now making the graphs...")
        print("")

        if CUMULATIVE_GRAPHS:
            ax = plotter(fig_names=(
                '_1_distributions_same_diff_distances_cum', '_2_distributions_same_diff_distances_with_norm_cum'),
                cumulative=1, same_face_distances_df=same_face_distances_df,
                different_face_distances_df=different_face_distances_df, comparison_counter=comparison_counter,
                file_str_prefix=file_str_prefix, lists_of_images=lists_of_images)

        else:
            ax = plotter(
                fig_names=('_1_distributions_same_diff_distances', '_2_distributions_same_diff_distances_with_norm'),
                cumulative=False, same_face_distances_df=same_face_distances_df,
                different_face_distances_df=different_face_distances_df, comparison_counter=comparison_counter,
                file_str_prefix=file_str_prefix, lists_of_images=lists_of_images)

        roc_auc(same_face_distances_df, different_face_distances_df, ax, comparison_counter, lists_of_images,
                file_str_prefix)

        return graph_start_time


def combine_face_images(face_images_df, file_str_prefix, image_note_str):
    """

    :param face_images_df:
    :param file_str_prefix:
    :param image_note_str:
    :return:
    """
    #todo: delete
    #face_images_df['path1'] = face_images_df['path1'].str.replace('../2018-11_Lookalike_finder/lfw', lfw_path)
    #face_images_df['path2'] = face_images_df['path2'].str.replace('../2018-11_Lookalike_finder/lfw', lfw_path)

    face_images_df['path1'] = face_images_df['path1'].str.replace(r'\\', '/')
    face_images_df['path2'] = face_images_df['path2'].str.replace(r'\\', '/')

    i = 0
    for row in face_images_df.itertuples():

        image1 = cv2.imread(row[3])
        image2 = cv2.imread(row[2])

        image1 = cv2.resize(image1, (250, 250))
        image2 = cv2.resize(image2, (250, 250))

        cv2.putText(image1, str(i + 1), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        this_pair = np.concatenate([image1, image2], axis=1)

        if i == 0:
            all_images = this_pair

        else:
            all_images = np.concatenate([all_images, this_pair], axis=0)

        i += 1

    cv2.imwrite(os.path.join(os.path.join(file_str_prefix + image_note_str)),
                all_images)


def plot_first_names_wordcloud(file_str_prefix, lists_of_images):
    """

    :param lfw_path:
    :return:
    """
    print("\n" + datetime.datetime.now().strftime("%Y_%m_%d__%H:%M:%S") + " Now making names wordcloud...")

    names = [name.lower().capitalize() for name in lists_of_images['all_peoples_first_names']]

    names_string = " ".join(names)
    wordcloud = WordCloud(max_words=1000, width=1800, height=600, collocations=False).generate(names_string)

    filename = file_str_prefix + '_10_names_wordcloud.jpg'
    imsave(os.path.join(results_directory, filename), wordcloud)

def count_images_lfw(lfw_path):
    """

    :param lfw_path:
    :return:
    """
    folders = [folder for folder in os.listdir(lfw_path) if '.' not in folder]
    files = [file for file in [os.listdir(os.path.join(lfw_path, folder)) for folder in folders] if '.db' not in file]
    files = [item for items in files for item in items]

    return len(folders), len(files)
