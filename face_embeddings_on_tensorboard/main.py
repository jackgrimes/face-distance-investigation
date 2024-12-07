from utils import create_sprite, encode_faces, run_tensorboard, set_up_tensorboard

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
