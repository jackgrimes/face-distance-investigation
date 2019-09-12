import datetime
import os

from scipy.misc import imsave
from wordcloud import WordCloud

from configs import base_directory, results_directory


def plot_first_names_wordcloud(base_directory):
    """

    :param base_directory:
    :return:
    """
    names = [item for sublist in [x.split("_") for x in os.listdir(base_directory)] for item in sublist]
    names = [name.lower() for name in names]
    names = [name.capitalize() for name in names]

    person_count = len(os.listdir(base_directory))

    filename = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M") + '_wordcloud_from_' \
               + str(person_count) + "_names.jpg"

    names_string = " ".join(names)

    wordcloud = WordCloud(max_words=1000, width=1800, height=600).generate(names_string)

    imsave(os.path.join(os.path.join(results_directory, 'names_wordcloud'), filename), wordcloud)


plot_first_names_wordcloud(base_directory)
