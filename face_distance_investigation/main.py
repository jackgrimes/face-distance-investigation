import datetime

from configs import (
    CUMULATIVE_GRAPHS,
    IMAGES_TO_EXCLUDE,
    doing_graphs,
    doing_precision_recall,
    lfw_path,
)
from utils import (
    all_graphs,
    combine_face_images,
    encodings_builder,
    encodings_comparer,
    get_number_faces_to_scan,
    output_most_similar_different_people_and_most_different_same_faces,
    plot_first_names_wordcloud,
    precision_recall,
    run_outputs,
)


def main():
    overall_start_time = datetime.datetime.now()

    # Allow user to compare only a subset of the faces
    (
        number_of_people_to_scan,
        attempting_all,
        file_str_prefix,
        peoples_faces_to_scan,
    ) = get_number_faces_to_scan(lfw_path, overall_start_time)

    # Build up encodings dataset
    all_encodings, encodings_start_time, lists_of_images = encodings_builder(
        lfw_path, number_of_people_to_scan, peoples_faces_to_scan, IMAGES_TO_EXCLUDE
    )

    # Compare the encodings
    (
        same_face_distances_df,
        different_face_distances_df,
        comparisons_start_time,
        comparison_counter,
    ) = encodings_comparer(all_encodings)

    # Make graphs
    graph_start_time = all_graphs(
        same_face_distances_df,
        different_face_distances_df,
        comparison_counter,
        lists_of_images,
        file_str_prefix,
        doing_graphs,
        CUMULATIVE_GRAPHS,
    )

    # Calculate precision and recall
    precision_recall_start_time = precision_recall(
        same_face_distances_df,
        different_face_distances_df,
        file_str_prefix,
        doing_precision_recall,
    )

    # Find lookalikes and different-looking images of same person
    (
        different_face_distances_df_sorted,
        same_face_distances_df_sorted,
    ) = output_most_similar_different_people_and_most_different_same_faces(
        different_face_distances_df, same_face_distances_df, file_str_prefix
    )

    # Image of lookalikes etc
    combine_face_images(
        different_face_distances_df_sorted, file_str_prefix, "_8_lookalikes.jpg"
    )
    combine_face_images(
        same_face_distances_df_sorted,
        file_str_prefix,
        "_9_different_looking_same_people.jpg",
    )

    # First names wordcloud
    plot_first_names_wordcloud(file_str_prefix, lists_of_images)

    # Write out timings and info about images that failed
    run_outputs(
        attempting_all,
        overall_start_time,
        encodings_start_time,
        comparisons_start_time,
        graph_start_time,
        precision_recall_start_time,
        file_str_prefix,
        lists_of_images,
    )


if __name__ == "__main__":
    main()
