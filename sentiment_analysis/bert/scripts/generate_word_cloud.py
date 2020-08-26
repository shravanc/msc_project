from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd

import os
train_base_dir = "/home/shravan/Downloads/train/"
valid_base_dir = "/home/shravan/Downloads/valid/"


def load_datasets():
    train_df = pd.DataFrame()
    for name in os.listdir(train_base_dir):
        file_path = os.path.join(train_base_dir, name)
        train_df = pd.concat([train_df,
                              pd.read_csv(file_path, sep=',', names=["sentences", "polarity"])],
                             ignore_index=True
                             )

    valid_df = pd.DataFrame()
    for name in os.listdir(valid_base_dir):
        file_path = os.path.join(valid_base_dir, name)
        valid_df = pd.concat([valid_df,
                              pd.read_csv(file_path, sep=',', names=["sentences", "polarity"])],
                             ignore_index=True
                             )

    return train_df, valid_df


df, test = load_datasets()
print(df.head())
print(df['sentences'])

comment_words = ''

for val in df.sentences:

    # typecaste each val to string
    val = str(val)

    # split the value
    tokens = val.split()

    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()

    comment_words += " ".join(tokens)+" "

wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                min_font_size = 10).generate(comment_words)

# plot the WordCloud image
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)

plt.show()
