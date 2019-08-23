import datetime

from configs import doing_graphs, doing_precision_recall, base_directory
from utils import encodings_builder, get_number_faces_to_scan, encodings_comparer, \
    precision_recall, run_outputs, output_most_similar_different_people_and_most_different_same_faces, \
    all_graphs, combine_face_images


def main():
    overall_start_time = datetime.datetime.now()

    # Allow user to compare only a subset of the faces
    image_no_max, attempting_all, person_count, file_str_prefix = get_number_faces_to_scan(base_directory,
                                                                                           overall_start_time)

    # Build up encodings dataset
    (all_encodings, encodings_start_time, counters) = encodings_builder(base_directory, image_no_max, attempting_all)

    # Compare the encodings
    (same_face_distances_df,
     different_face_distances_df,
     comparisons_start_time,
     comparison_counter) = encodings_comparer(
        all_encodings)

    # Make graphs
    graph_start_time = all_graphs(same_face_distances_df, different_face_distances_df, comparison_counter, counters,
                                  file_str_prefix, doing_graphs)

    # Calculate precision and recall
    precision_recall_start_time = precision_recall(same_face_distances_df, different_face_distances_df, file_str_prefix,
                                                   doing_precision_recall)

    # Find lookalikes and different-looking images of same person
    (different_face_distances_df_sorted,
     same_face_distances_df_sorted) = output_most_similar_different_people_and_most_different_same_faces(
        different_face_distances_df,
        same_face_distances_df, file_str_prefix)

    # Image of lookalikes etc

    combine_face_images(different_face_distances_df_sorted, file_str_prefix, '_10_lookalikes.jpg')
    combine_face_images(same_face_distances_df_sorted, file_str_prefix, '_11_different_looking_same_people.jpg')

    # Write out timings and info about images that failed
    run_outputs(attempting_all, overall_start_time,
                encodings_start_time, comparisons_start_time, graph_start_time, precision_recall_start_time,
                file_str_prefix, counters
                )


if __name__ == "__main__":
    main()
