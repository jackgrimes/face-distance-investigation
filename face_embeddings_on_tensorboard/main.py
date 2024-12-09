import logging
from logging import getLogger

from configs import (
    CREATING_SPRITE,
    ENCODING_LFW_FACES,
    ENCODING_OTHER_IMAGES,
    IMAGE_LIMIT,
    RUNNING_TENSORBOARD,
    SETTING_UP_TENSORBOARD,
)
from utils import (
    create_sprite,
    encode_faces,
    encode_other_images,
    run_tensorboard,
    set_up_tensorboard,
)

logger = getLogger(__name__)
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

if __name__ == "__main__":
    logger.info("Running main.py")

    if ENCODING_LFW_FACES:
        encode_faces(IMAGE_LIMIT)

    if ENCODING_OTHER_IMAGES:
        encode_other_images()
        CREATING_SPRITE = True

    if CREATING_SPRITE:
        create_sprite()

    if SETTING_UP_TENSORBOARD:
        set_up_tensorboard()

    if RUNNING_TENSORBOARD:
        run_tensorboard()
