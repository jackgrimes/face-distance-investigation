import os

doing_graphs = True
doing_precision_recall = True

# Provide path to faces dataset
lfw_path = r"C:\dev\data\lfw"

# Results directory
results_directory = r"C:\dev\data\face_distance_investigation"

# Define allowed extensions for the images to scan
ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg", "gif"])

# Cumulative or normal graphs of face distances?
CUMULATIVE_GRAPHS = False

# How many lookalikes and different-looking images of same person to include in outputs
N_LOOKALIKES_AND_DIFFERENT_LOOKING_SAME_PEOPLE_TO_INCLUDE = 10

# Any images to exclude, because they cause problems - for example because they are duplicated with different names
# or because the labelled person's face doesn't get recognised by face_recognition
IMAGES_TO_EXCLUDE = [
    r"Ricky_Ray\Ricky_Ray_0001.jpg",
    r"Bart_Hendricks\Bart_Hendricks_0001.jpg",
    r"Raul_Ibanez\Raul_Ibanez_0001.jpg",
    r"Carlos_Beltran\Carlos_Beltran_0001.jpg",
    r"James_Becker\James_Becker_0001.jpg",
    r"Franklin_Damann\Franklin_Damann_0001.jpg",
    r"Emmy_Rossum\Emmy_Rossum_0001.jpg",
    r"Gabrielle_Rose\Gabrielle_Rose_0001.jpg",
    r"Andrew_Caldecott\Andrew_Caldecott_0001.jpg",
    r"Anja_Paerson\Anja_Paerson_0001.jpg",
    r"Vecdi_Gonul\Vecdi_Gonul_0001.jpg",
    r"Abdullah_Gul\Abdullah_Gul_0012.jpg",
    r"Morgan_Hentzen\Morgan_Hentzen_0001.jpg",
    r"Colin_Powell\Colin_Powell_0206.jpg",
    r"Tung_Chee-hwa\Tung_Chee-hwa_0001.jpg",
    r"Mahmoud_Abbas\Mahmoud_Abbas_0021.jpg",
    r"Hans_Blix\Hans_Blix_0026.jpg",
    r"Mahathir_Mohamad\Mahathir_Mohamad_0009.jpg",
    r"Michael_Schumacher\Michael_Schumacher_0008.jpg",
    r"Recep_Tayyip_Erdogan\Recep_Tayyip_Erdogan_0004.jpg",
    r"Janica_Kostelic\Janica_Kostelic_0001.jpg",
]
IMAGES_TO_EXCLUDE = [os.path.join(lfw_path, file) for file in IMAGES_TO_EXCLUDE]

# Some people are mislabelled and these are examples of people who are actually the same person
ACTUALLY_SAME_PEOPLE = {
    "Felipe_De_Borbon": "Prince_Felipe",
    "Noer_Muis": "Noer_Moeis",
    "Takahiro_Mori": "Shinya_Taniguchi",
    "Yingfan_Wang": "Wang_Yingfan",
    "Sung_Hong_Choi": "Choi_Sung-hong",
    "Dai_Chul_Chyung": "Chyung_Dai-chul",
    "John_Burnett": "John_Barnett",
    "Hassan_Wirajuda": "Hasan_Wirayuda",
    "Wen_Jiabao": "Hu_Jintao",
    "Eduardo_Duhalde": "Carlos_Ruckauf",
    "Wang_Nan": "Nan_Wang",
    "Tia_Mowry": "Tamara_Mowry",
}
