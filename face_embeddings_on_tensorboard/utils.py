import os

import cv2
import face_recognition
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from configs import data_path, lfw_path
from tensorboard.plugins import projector
from tqdm import tqdm

tf.disable_v2_behavior()

LOG_DIR = os.path.join(data_path, "tensorboard_logs")
IMAGE_SIZE = (64, 64)
CHECKPOINT_FILE = os.path.join(LOG_DIR, "features.ckpt")
METADATA_FILE = os.path.join(LOG_DIR, "metadata.tsv")
SPRITES_FILE = os.path.join(LOG_DIR, "sprites.jpg")


def encode_faces(limit=None):
    all_encodings = pd.DataFrame()
    metadata = pd.DataFrame()
    counter = 0

    people = os.listdir(lfw_path)

    if limit is None:
        limit = len(people)

    for person in tqdm(people[0:limit]):
        files = os.listdir(os.path.join(lfw_path, person))
        files = [file for file in files if file.endswith(".jpg")]
        if len(files) > 0:
            for file in files:
                image = face_recognition.load_image_file(
                    os.path.join(lfw_path, person, file)
                )
                encodings = face_recognition.face_encodings(image)
                if len(encodings) == 1:
                    encodings = encodings[0]
                    new_encodings = pd.DataFrame(encodings).T
                    new_encodings.index = [counter]
                    all_encodings = pd.concat([all_encodings, new_encodings])
                    metadata = pd.concat(
                        [
                            metadata,
                            pd.DataFrame(
                                {
                                    "name": person.replace("_", " "),
                                    "path": os.path.join(lfw_path, person, file),
                                },
                                index=[counter],
                            ),
                        ]
                    )
                    counter += 1
                    if counter > limit:
                        break
                if counter > limit:
                    break
        if counter > limit:
            break

    all_encodings.to_csv(
        os.path.join(data_path, "tensorboard_logs", "all_encodings.tsv"),
        sep="\t",
        index=False,
        header=False,
    )
    metadata.to_csv(
        os.path.join(data_path, "tensorboard_logs", "metadata.tsv"),
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


def create_sprite(image_files):
    # Max sprite size is 8192 x 8192 so this max samples makes visualization easy
    MAX_NUMBER_SAMPLES = 8191

    img_data = []
    for img in image_files[:MAX_NUMBER_SAMPLES]:
        input_img = cv2.imread(img)
        input_img_resize = cv2.resize(input_img, IMAGE_SIZE)
        img_data.append(input_img_resize)

    img_data = np.array(img_data)

    sprite = combine_images(img_data)
    cv2.imwrite(SPRITES_FILE, sprite)


def create_sprite_wrapper():
    metadata = pd.read_csv(
        os.path.join(data_path, "tensorboard_logs", "metadata.tsv"), sep="\t"
    )
    paths = metadata["path"].tolist()
    create_sprite(image_files=paths)


def visualize_encodings():

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
        saver.save(sess, CHECKPOINT_FILE)

        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = features.name
        embedding.metadata_path = METADATA_FILE

        # This adds the sprite images
        embedding.sprite.image_path = SPRITES_FILE
        embedding.sprite.single_image_dim.extend(IMAGE_SIZE)
        projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)
