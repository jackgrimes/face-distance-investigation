import datetime
import os

from configs import doing_graphs, doing_precision_recall, base_directory
from utils import encodings_builder, get_number_faces_to_scan, encodings_comparer, \
    plotter, roc_auc, precision_recall, run_outputs, output_most_similar_different_people_and_most_different_same_faces


def main():
    overall_start_time = datetime.datetime.now()

    # Allow user to compare only a subset of the faces

    image_no_max, attempting_all, person_count = get_number_faces_to_scan(base_directory)

    # Build up encodings dataset

    encodings_start_time = datetime.datetime.now()
    start_time = datetime.datetime.now()

    # Build up the encodings

    all_encodings, image_no, person_no, image_attempt_no, failed_attempts, images_without_faces = encodings_builder(
        base_directory, image_no_max, image_no_max, attempting_all, encodings_start_time, start_time)

    # Compare the encodings

    comparisons_start_time = datetime.datetime.now()
    print("\n" + comparisons_start_time.strftime("%Y_%m_%d %H:%M:%S") + " Doing the comparisons..." + "\n")

    same_face_distances, different_face_distances, comparison_counter, different_face_distances_df, same_face_distances_df = encodings_comparer(
        all_encodings)

    #
    ##GRAPHS
    #

    graph_start_time = datetime.datetime.now()

    if attempting_all:
        file_str_prefix = os.path.join(r"C:\dev\data\face_distance_investigation",
                                       graph_start_time.strftime("%Y_%m_%d %H_%M_%S_") + "_attempting_all_images_")
    else:
        file_str_prefix = os.path.join(r"C:\dev\data\face_distance_investigation",
                                       graph_start_time.strftime("%Y_%m_%d %H_%M_%S_") + "_attempting_" + str(
                                           image_no_max) + "_images_")

    if doing_graphs:
        print(datetime.datetime.now().strftime("%Y_%m_%d %H:%M:%S") + " Now making the graphs...")
        print("")

        # Non-cumulative histograms

        ax = plotter(
            fig_names=('_1_distributions_same_diff_distances', '_2_distributions_same_diff_distances_with_norm'),
            cumulative=False, same_face_distances=same_face_distances,
            different_face_distances=different_face_distances, comparison_counter=comparison_counter,
            image_no=image_no, person_no=person_no, file_str_prefix=file_str_prefix)

        # Cumulative histograms

        ax = plotter(fig_names=(
            '_3_distributions_same_diff_distances_cum', '_4_distributions_same_diff_distances_with_norm_cum'),
            cumulative=1, same_face_distances=same_face_distances,
            different_face_distances=different_face_distances, comparison_counter=comparison_counter,
            image_no=image_no, person_no=person_no, file_str_prefix=file_str_prefix)

        # ROC and AUC

        roc_auc(same_face_distances, different_face_distances, ax, comparison_counter, person_no, image_no,
                file_str_prefix)


    #
    ##PRECISION, RECALL
    #

    precision_recall_start_time = datetime.datetime.now()

    if doing_precision_recall:
        print(datetime.datetime.now().strftime("%Y_%m_%d %H:%M:%S") + " Now doing precision and recall calculations...")
        print("")

        precision_recall(same_face_distances, different_face_distances, file_str_prefix)

    #
    ## LOOKALIKES AND DIFFERENT-LOOKING PHOTOS OF SAME PERSON
    #

    output_most_similar_different_people_and_most_different_same_faces(different_face_distances_df,
                                                                       same_face_distances_df, file_str_prefix)

    #
    ##RUN OUTPUTS
    #

    run_outputs(attempting_all, images_without_faces, image_attempt_no, failed_attempts, image_no, overall_start_time,
                encodings_start_time, comparisons_start_time, graph_start_time, precision_recall_start_time,
                file_str_prefix
                )


if __name__ == "__main__":
    main()
