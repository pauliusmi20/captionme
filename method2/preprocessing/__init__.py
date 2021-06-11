import pandas as pd
from method2 import FLICKR_PATH_CSV, FLICKR_PATH_DIR, GLOVE_PATH

class PreprocessingM2():
    def __init__(self):
        self.flickr_csv = pd.read_csv(FLICKR_PATH_CSV, error_bad_lines=False, sep="|")

        self.flickr_csv.columns = [col.strip() for col in self.flickr_csv.columns]

        # We notice that there are spaces in the leading of cells : let's removing them
        for col in self.flickr_csv.columns:
            self.flickr_csv[col] = self.flickr_csv[col].apply(lambda x: str(x).lstrip())

        # Put all the comments into lowercase
        self.flickr_csv['comment'] = self.flickr_csv['comment'].apply(lambda x: x.lower())

        # Add new column 'image_name_process' by concatenating 'image_name' without the extension '.jpg' and the 'comment_number'
        self.flickr_csv["image_name_process"] = self.flickr_csv.apply(
            lambda row: row["image_name"].split(".")[0] + "." + row["comment_number"], axis=1)

        # Sort flickr_csv by 'image_name_process'
        self.flickr_csv = self.flickr_csv.sort_values("image_name_process")
        # Keep only 8090 lines
        self.flickr_csv = self.flickr_csv.loc[:8000 * 5 - 1]
        self.flickr_csv.head()
        self.flickr_csv.tail()

        """Verifying if all the `image_name` in the `Dataframe flickr_csv` have an image in the folder `FLICKR_PATH_DIR`."""

        # Verfify if an image_name corresponds to an existing image
        import os
        image_name_in_folder = os.listdir(FLICKR_PATH_DIR)

        cpt_index_images_name_to_drop = 0

        for index, row in self.flickr_csv.iterrows():
            if row['image_name'] not in image_name_in_folder:
                cpt_index_images_name_to_drop += 1

        if cpt_index_images_name_to_drop == 0:
            print("All image_name values in the dataset has an image in the folder !")
        else:
            print("Some image_name values in the dataset hasn't an image in the folder !")

        """Verifying if all the `image_name` in the `Dataframe flickr_csv` have 5 comments."""

        # Verify how many comments are there for every image
        self.flickr_gbcount = self.flickr_csv.groupby("image_name").count()

        # Count the number of 'image_name_process' after groupby and count inferior to 5
        cpt = 0

        for index, row in self.flickr_gbcount.iterrows():
            if row["image_name_process"] == 5:
                cpt += 1

        if len(self.self.flickr_gbcount) == cpt:
            print("All the images have 5 comments !")
        else:
            print(
                "Some images have less that 5 comments !\nThis must be due to the reduction number to 1 of comments per image...")

        self.flickr_gbcount.head()

        """<u>Remove punctuation</u> :
        
        Punctuation can provide grammatical context to a sentence which supports our understanding. But for our vectorizer which counts the number of words and not the context, it does not add value, so we remove all special characters. eg: How are you?->How are you
        """

        import string


        # Function to remove the punctuation
        def remove_punctuation(text):
            """
            Takes a text and remove its punctuation.
            """
            text_without_punct = "".join([char for char in text if char not in string.punctuation])

            return text_without_punct


        self.flickr_csv["comment_text_clean"] = self.flickr_csv["comment"].apply(lambda x: remove_punctuation(str(x)))

        self.flickr_csv.head()

        self.flickr_csv["comment"][0]

        """Next, we create a dictionary named “descriptions” which contains the name of the image as keys and a list of the 5 captions for the corresponding image as values."""

        descriptions = dict()

        for index, row in self.flickr_csv.iterrows():
            image_name = row['image_name'].split(".")[0]
            desc = row["comment_text_clean"]
            if image_name not in descriptions:
                descriptions[image_name] = []

            descriptions[image_name].append(desc)

        descriptions["1000092795"]

        """Next, we create a vocabulary of all the unique words present across all the 8000*5 (i.e. 40000) image captions in the data set. We have 8828 unique words across all the 40000 image captions."""

        vocabulary = set()
        for key in descriptions.keys():
            [vocabulary.update(d.split()) for d in descriptions[key]]
        print('Original Vocabulary Size: %d' % len(vocabulary))

        """Now let’s save the image id’s and their new cleaned captions in the same format as the token.txt file:-"""

        lines = list()
        for key, desc_list in descriptions.items():
            for desc in desc_list:
                lines.append(key + ' ' + desc)
        new_descriptions = '\n'.join(lines)

        new_descriptions.split("\n")[0]

        """Next, we load all the 6000 training image id’s in a variable train."""

        # Contains all the images_name without the ".png" for the training set
        self.train = set([imageName.split(".")[0] for imageName in self.flickr_csv["image_name"].loc[:30000 - 1]])
        print("Varibale`train` has : {} values".format(len(self.train)))

        # Contains all the images_name without the ".png" for the testing set
        self.test = set([imageName.split(".")[0] for imageName in self.flickr_csv["image_name"].loc[30000:40000 - 1]])
        print("Varibale`train` has : {} values".format(len(self.test)))

        """Now we save all the training and testing images in train_img and test_img lists respectively. We verify again if all the `image_name` of whole train and test dataset are available on the image directory and thus verify if all the `image_name` are mapped in the image directory."""

        import glob

        images = glob.glob(FLICKR_PATH_DIR + '*.jpg')
        self.train_images = self.train
        self.train_img = []

        for image in images:
            if image[len(FLICKR_PATH_DIR):].split(".")[0] in self.train_images:
                self.train_img.append(image)

        self.test_images = self.test
        test_img = []
        for image in images:
            if image[len(FLICKR_PATH_DIR):].split(".")[0] in self.test_images:
                test_img.append(image)

        print("Number of training images : {}".format(len(self.train_img)))
        print("Number of test images : {}".format(len(test_img)))

        """Now, we load the descriptions of the training images into a dictionary. However, we will add two tokens in every caption, which are ‘startseq’ and ‘endseq’."""

        self.train_descriptions = dict()
        for line in new_descriptions.split('\n'):
            tokens = line.split()
            image_id, image_desc = tokens[0], tokens[1:]
            if image_id in self.train:
                if image_id not in self.train_descriptions:
                    self.train_descriptions[image_id] = list()
                desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
                self.train_descriptions[image_id].append(desc)

        self.train_descriptions[list(self.train_descriptions.keys())[0]]

        """Create a list of all the training captions."""

        all_train_captions = []
        for key, val in self.train_descriptions.items():
            for cap in val:
                all_train_captions.append(cap)

        all_train_captions[:5]

        """
        To make our model more robust we will reduce our vocabulary to only those words which occur at least 10 times in the entire corpus."""

        word_count_threshold = 10
        word_counts = {}
        nsents = 0
        for sent in all_train_captions:
            nsents += 1
            for w in sent.split(' '):
                word_counts[w] = word_counts.get(w, 0) + 1
        vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]

        print('Vocabulary = %d' % (len(vocab)))

        """Now we create two dictionaries to map words to an index and vice versa. Also, we append 1 to our vocabulary since we append 0’s to make all captions of equal length."""

        self.ixtoword = {}
        self.wordtoix = {}
        ix = 1
        for w in vocab:
            self.wordtoix[w] = ix
            self.ixtoword[ix] = w
            ix += 1

        vocab_size = len(self.ixtoword) + 1

        print("Size of vocabulary : {}".format(vocab_size))
        print("`ixtoword` = {}".format(len(self.ixtoword)))
        print("`wordtoix` = {}".format(len(self.wordtoix)))

        print("ixtoword -> {} : {}".format(list(self.ixtoword.keys())[0], self.ixtoword[list(self.ixtoword.keys())[0]]))
        print("wordtoix -> {} : {}".format(list(self.wordtoix.keys())[0], self.wordtoix[list(self.wordtoix.keys())[0]]))

        """Hence now our total vocabulary size is 2101.
        We also need to find out what the max length of a caption can be since we cannot have captions of arbitrary length.
        """

        all_desc = list()
        for key in self.train_descriptions.keys():
            [all_desc.append(d) for d in self.train_descriptions[key]]
        self.lines = all_desc
        self.max_length = max(len(d.split()) for d in lines)

        print('Description Length: %d' % self.max_length)

        """## Glove Embeddings
        
        Word vectors map words to a vector space, where similar words are clustered together and different words are separated. The advantage of using Glove over Word2Vec is that GloVe does not just rely on the local context of words but it incorporates global word co-occurrence to obtain word vectors.
        
        The basic premise behind Glove is that we can derive semantic relationships between words from the co-occurrence matrix. For our model, we will map all the words in our 38-word long caption to a 200-dimension vector using Glove.
        
        ```python
          dict(string : list(float32))  glov6d
        ```
        """
        import os
        import numpy as np

        self.embeddings_index = {}
        f = open(os.path.join(GLOVE_PATH, 'glove.6B.200d.txt'), encoding="utf-8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs

        print("Embeddings index \n {} : {}".format(list(self.embeddings_index.keys())[0],
                                                   self.embeddings_index[list(self.embeddings_index.keys())[0]]))

        """Next, we make the matrix of shape (2100,200) consisting of our vocabulary and the 200-d vector."""

        self.embedding_dim = 200
        self.embedding_matrix = np.zeros((vocab_size, self.embedding_dim))
        for word, i in self.wordtoix.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector

