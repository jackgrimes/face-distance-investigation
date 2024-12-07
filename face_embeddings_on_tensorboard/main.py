from utils import create_sprite_wrapper, encode_faces, visualize_encodings

ENCODING_FACES = True
CREATING_SPRITE = True
VISUALIZING_ENCODINGS = True

if ENCODING_FACES:
    encode_faces(limit=30)

if CREATING_SPRITE:
    create_sprite_wrapper()

if VISUALIZING_ENCODINGS:
    visualize_encodings()
