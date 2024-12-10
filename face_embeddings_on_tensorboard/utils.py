import os
import re
import subprocess
from logging import getLogger

import cv2
import face_recognition
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from configs import (
    CHECKPOINT_FILE,
    IMAGE_SIZE,
    METADATA_FILE,
    N_ROWS_PER_FILE,
    data_path,
    formats,
    lfw_path,
    other_images_path,
)
from tensorboard.plugins import projector
from tqdm import tqdm

tf.disable_v2_behavior()

logger = getLogger(__name__)


def take_most_central_face(encodings, image, file_path):
    # If no other images for this person, take the most central face

    # locations are top, right, bottom, left
    face_locations = face_recognition.face_locations(image)

    im = cv2.imread(file_path)
    height, width, _ = im.shape

    distances = []

    for location in face_locations:
        top, right, bottom, left = location
        y, x = (top + bottom) / 2, (right + left) / 2
        y_offset = y - (height / 2)
        x_offset = x - (width / 2)

        distance_from_centre = np.sqrt(y_offset**2 + x_offset**2)

        distances.append(distance_from_centre)

    i = np.argmin(distances)
    encodings = [encodings[i]]
    return encodings


def compare_with_other_images_this_person(encodings, other_images_this_person):
    # If other faces for this person, take the face that's the closest match with the other faces
    other_faces_this_person = []
    for other_image in other_images_this_person:
        other_image_loaded = face_recognition.load_image_file(other_image)
        other_faces_this_person.extend(
            face_recognition.face_encodings(other_image_loaded)
        )
    if len(other_faces_this_person) > 0:
        results = []
        for face in encodings:
            d = face_recognition.face_distance(other_faces_this_person, face)
            average_d = d.mean()
            results.append(average_d)
        encodings = [encodings[np.argmin(results)]]

    return encodings


def get_other_face_encodings_this_person(other_images_this_person):
    other_faces_this_person = []
    for other_image in other_images_this_person:
        other_image_loaded = face_recognition.load_image_file(other_image)
        other_faces_this_person.extend(
            face_recognition.face_encodings(other_image_loaded)
        )
        return other_faces_this_person


def select_best_face(encodings, this_person_folder, file_path, image):
    this_person_images = [
        os.path.join(this_person_folder, f)
        for f in os.listdir(this_person_folder)
        if f.endswith(".jpg")
    ]
    other_images_this_person = [i for i in this_person_images if i != file_path]

    if len(other_images_this_person) == 0:
        encodings = take_most_central_face(encodings, image, file_path)
        return encodings

    if len(other_images_this_person) > 0:
        other_face_encodings_this_person = get_other_face_encodings_this_person(
            other_images_this_person
        )
        if len(other_face_encodings_this_person) == 0:
            encodings = take_most_central_face(encodings, image, file_path)
            return encodings
        else:
            encodings = compare_with_other_images_this_person(
                encodings, other_images_this_person
            )
            return encodings


def encode_faces(limit=None):
    old_files = sorted(
        [
            f
            for f in os.listdir(os.path.join(data_path, "tensorboard_logs"))
            if (f.startswith("lfw_encodings") or f.startswith("lfw_metadata"))
        ]
    )

    for f in old_files:
        os.remove(os.path.join(data_path, "tensorboard_logs", f))

    all_lfw_encodings = pd.DataFrame()
    lfw_metadata = pd.DataFrame()
    counter = 0
    file_counter = 0

    logger.info("Encoding faces")

    logger.info("Scanning for files")

    all_file_paths = []
    for dirname, subdirs, files in os.walk(lfw_path):
        for fname in files:
            full_path = os.path.join(dirname, fname)
            if fname.endswith(".jpg"):
                all_file_paths.append(full_path)

    if limit is not None:
        all_file_paths = all_file_paths[0:limit]

    logger.info("Getting face encodings from lfw files")
    for file_path in tqdm(all_file_paths):
        this_person_folder = os.path.dirname(file_path)
        person = os.path.basename(this_person_folder).replace("_", " ")

        image = face_recognition.load_image_file(
            os.path.join(lfw_path, person, file_path)
        )
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 1:
            encodings = select_best_face(
                encodings, this_person_folder, file_path, image
            )

        if len(encodings) == 1:
            encodings = encodings[0]
            new_encodings = pd.DataFrame(encodings).T
            new_encodings.index = [counter]
            all_lfw_encodings = pd.concat([all_lfw_encodings, new_encodings])
            lfw_metadata = pd.concat(
                [
                    lfw_metadata,
                    pd.DataFrame(
                        {
                            "name": person.replace("_", " "),
                            "path": os.path.join(lfw_path, person, file_path),
                        },
                        index=[counter],
                    ),
                ]
            )
            counter += 1

            if counter % N_ROWS_PER_FILE == 0:
                all_lfw_encodings.to_csv(
                    os.path.join(
                        data_path,
                        "tensorboard_logs",
                        f"lfw_encodings_{file_counter}.tsv",
                    ),
                    sep="\t",
                    index=False,
                    header=False,
                )
                lfw_metadata.to_csv(
                    os.path.join(
                        data_path,
                        "tensorboard_logs",
                        f"lfw_metadata_{file_counter}.tsv",
                    ),
                    sep="\t",
                    index=False,
                )
                file_counter += 1
                all_lfw_encodings = pd.DataFrame()
                lfw_metadata = pd.DataFrame()

    all_lfw_encodings.to_csv(
        os.path.join(
            data_path, "tensorboard_logs", f"lfw_encodings_{file_counter}.tsv"
        ),
        sep="\t",
        index=False,
        header=False,
    )
    lfw_metadata.to_csv(
        os.path.join(data_path, "tensorboard_logs", f"lfw_metadata_{file_counter}.tsv"),
        sep="\t",
        index=False,
    )

    all_lfw_encodings = pd.DataFrame()
    lfw_metadata = pd.DataFrame()

    encodings_files = sorted(
        [
            f
            for f in os.listdir(os.path.join(data_path, "tensorboard_logs"))
            if (f.endswith(".tsv") and f.startswith("lfw_encodings_"))
        ]
    )

    for f in encodings_files:
        df = pd.read_csv(
            os.path.join(data_path, "tensorboard_logs", f), sep="\t", header=None
        )
        all_lfw_encodings = pd.concat([all_lfw_encodings, df])
    all_lfw_encodings.to_csv(
        os.path.join(data_path, "tensorboard_logs", "lfw_encodings.tsv"),
        sep="\t",
        index=False,
        header=False,
    )
    for f in encodings_files:
        os.remove(os.path.join(data_path, "tensorboard_logs", f))

    lfw_metadata_files = sorted(
        [
            f
            for f in os.listdir(os.path.join(data_path, "tensorboard_logs"))
            if (f.endswith(".tsv") and f.startswith("lfw_metadata_"))
        ]
    )
    for f in lfw_metadata_files:
        df = pd.read_csv(os.path.join(data_path, "tensorboard_logs", f), sep="\t")
        lfw_metadata = pd.concat([lfw_metadata, df])
    lfw_metadata.to_csv(
        os.path.join(data_path, "tensorboard_logs", "lfw_metadata.tsv"),
        sep="\t",
        index=False,
    )
    for f in lfw_metadata_files:
        os.remove(os.path.join(data_path, "tensorboard_logs", f))


def encode_other_images():
    old_files = sorted(
        [
            f
            for f in os.listdir(os.path.join(data_path, "tensorboard_logs"))
            if (f.startswith("other_encodings") or f.startswith("other_metadata"))
        ]
    )

    for f in old_files:
        os.remove(os.path.join(data_path, "tensorboard_logs", f))

    other_encodings = pd.DataFrame()
    other_metadata = pd.DataFrame()
    counter = 0

    logger.info("Encoding other faces")

    for filename in tqdm(
        [
            f
            for f in os.listdir(other_images_path)
            if any([f.endswith(format) for format in formats])
        ]
    ):
        file_path = os.path.join(other_images_path, filename)

        person = os.path.basename(file_path).replace("_", " ")

        image = face_recognition.load_image_file(file_path)

        encodings = face_recognition.face_encodings(image)

        encodings = encodings[0]
        new_encodings = pd.DataFrame(encodings).T
        new_encodings.index = [counter]
        other_encodings = pd.concat([other_encodings, new_encodings])
        other_metadata = pd.concat(
            [
                other_metadata,
                pd.DataFrame(
                    {
                        "name": re.sub(
                            r"[0-9]+",
                            "",
                            os.path.splitext(person)[0]
                            .replace("_", " ")
                            .title()
                            .strip(),
                        ),
                        "path": file_path,
                    },
                    index=[counter],
                ),
            ]
        )
        counter += 1

    other_encodings.to_csv(
        os.path.join(data_path, "tensorboard_logs", "other_encodings.tsv"),
        sep="\t",
        index=False,
        header=False,
    )
    other_metadata.to_csv(
        os.path.join(data_path, "tensorboard_logs", "other_metadata.tsv"),
        sep="\t",
        index=False,
    )

    all_encodings = pd.DataFrame()
    all_metadata = pd.DataFrame()

    encodings_files = ["lfw_encodings.tsv", "other_encodings.tsv"]
    for f in encodings_files:
        df = pd.read_csv(
            os.path.join(data_path, "tensorboard_logs", f), sep="\t", header=None
        )
        all_encodings = pd.concat([all_encodings, df])
    all_encodings.to_csv(
        os.path.join(data_path, "tensorboard_logs", "all_encodings.tsv"),
        sep="\t",
        index=False,
        header=False,
    )

    metadata_files = ["lfw_metadata.tsv", "other_metadata.tsv"]
    for f in metadata_files:
        df = pd.read_csv(os.path.join(data_path, "tensorboard_logs", f), sep="\t")
        all_metadata = pd.concat([all_metadata, df])
    all_metadata.to_csv(
        os.path.join(data_path, "tensorboard_logs", "all_metadata.tsv"),
        sep="\t",
        index=False,
    )


def combine_images(data):
    """
    Tile images into sprite image.
    Add any necessary padding
    """

    # For B&W or greyscale images
    if len(data.shape) == 3:
        data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n**2 - data.shape[0]), (0, 0), (0, 0), (0, 0))
    data = np.pad(data, padding, mode="constant", constant_values=0)

    # Tile images into sprite
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3, 4))

    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    return data


def create_sprite():
    logger.info("Creating sprite")
    all_metadata = pd.read_csv(
        os.path.join(data_path, "tensorboard_logs", "all_metadata.tsv"), sep="\t"
    )
    image_files = all_metadata["path"].tolist()

    # Max sprite size is 8192 x 8192
    n_images = len(image_files)
    n_rows = int(np.ceil(np.sqrt(n_images)))
    max_allowable_image_size = 8192 // n_rows

    if IMAGE_SIZE[0] > max_allowable_image_size:
        logger.info(
            f"Reducing image size down to {max_allowable_image_size} x {max_allowable_image_size} to keep sprite < 8192*8192 px"
        )
        image_size = (max_allowable_image_size, max_allowable_image_size)
    else:
        image_size = IMAGE_SIZE

    img_data = []
    for img in image_files:
        input_img = cv2.imread(img)
        input_img_resize = cv2.resize(input_img, image_size)
        img_data.append(input_img_resize)

    img_data = np.array(img_data)

    sprite = combine_images(img_data)

    sprite_files = [
        f
        for f in os.listdir(os.path.join(data_path, "tensorboard_logs"))
        if (f.endswith(".jpg") and f.startswith("sprites_"))
    ]
    for f in sprite_files:
        os.remove(os.path.join(data_path, "tensorboard_logs", f))

    cv2.imwrite(
        os.path.join(
            data_path,
            "tensorboard_logs",
            f"sprites_{image_size[0]}_{image_size[1]}.jpg",
        ),
        sprite,
    )


def set_up_tensorboard():
    logger.info("Setting up tensorboard")

    sprite_file = [
        f
        for f in os.listdir(os.path.join(data_path, "tensorboard_logs"))
        if (f.endswith(".jpg") and f.startswith("sprites_"))
    ][0]
    image_size = tuple([int(n) for n in re.findall("\d+", sprite_file)])

    df = pd.read_csv(
        os.path.join(data_path, "tensorboard_logs", "all_encodings.tsv"),
        sep="\t",
        header=None,
    )
    feature_vectors = df.to_numpy()

    features = tf.Variable(feature_vectors, name="features")

    with tf.compat.v1.Session() as sess:
        saver = tf.compat.v1.train.Saver([features])

        sess.run(features.initializer)
        saver.save(sess, os.path.join(data_path, "tensorboard_logs", CHECKPOINT_FILE))

        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = features.name
        embedding.metadata_path = os.path.join(
            data_path, "tensorboard_logs", METADATA_FILE
        )

        # This adds the sprite images
        embedding.sprite.image_path = os.path.join(
            data_path, "tensorboard_logs", sprite_file
        )
        embedding.sprite.single_image_dim.extend(image_size)
        projector.visualize_embeddings(
            tf.summary.FileWriter(os.path.join(data_path, "tensorboard_logs")), config
        )


def run_tensorboard():
    logger.info("Running tensorboard")
    subprocess.run(
        ["tensorboard", "--logdir", os.path.join(data_path, "tensorboard_logs")]
    )
