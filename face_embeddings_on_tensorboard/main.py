import logging
from logging import getLogger

from utils import create_sprite, encode_faces, run_tensorboard, set_up_tensorboard

logger = getLogger(__name__)
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

ENCODING_FACES = True
CREATING_SPRITE = True
SETTING_UP_TENSORBOARD = True
RUNNING_TENSORBOARD = True

if ENCODING_FACES:
    encode_faces()

if CREATING_SPRITE:
    create_sprite()

if SETTING_UP_TENSORBOARD:
    set_up_tensorboard()

if RUNNING_TENSORBOARD:
    run_tensorboard()
