lfw_path = "/path/to/lfw"
data_path = "path/to/data/folder/face-distance-investigation"
other_images_path = "/path/to/other/images/"

ENCODING_LFW_FACES = True
ENCODING_OTHER_IMAGES = True
CREATING_SPRITE = True
SETTING_UP_TENSORBOARD = True
RUNNING_TENSORBOARD = True

formats = [".jpg", ".jpeg", ".png"]

IMAGE_SIZE = (75, 75)
CHECKPOINT_FILE = "features.ckpt"
METADATA_FILE = "all_metadata.tsv"
N_ROWS_PER_FILE = 2000

IMAGE_LIMIT = None
