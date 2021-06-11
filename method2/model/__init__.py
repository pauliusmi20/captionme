
from keras import Input
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Embedding, Dense, Dropout
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from pickle import dump
from tensorflow.keras.utils import to_categorical, plot_model
from keras.callbacks import ModelCheckpoint
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np

import datetime
from method2 import FLICKR_SEED_PATH_DIR, FLICKR_PATH_CSV, FLICKR_PATH_DIR, FLICKR_SEED_PATH_DIR, FLICKR_PATH_DIR_8K_5C_METHOD2, FLICKR_PATH_DIR_8K_5C_METHOD2_MODEL  ,FLICKR_PATH_DIR_8K_5C_METHOD2_MODEL_CHECKPOINTS_DIR, GLOVE_PATH

class ModelM2:
    def __init__(self, preprocessedObj):
        self.preprocessedObj = preprocessedObj
        if not isinstance( preprocessedObj):
            print("Bad argument")
            return None
        self.model = InceptionV3(weights='imagenet')

        """We must remember that we do not need to classify the images here, we only need to extract an image vector for our images. Hence we remove the softmax layer from the inceptionV3 model."""

        model_new = Model(self.model.input, self.model.layers[-2].output)

        """Since we are using InceptionV3 we need to pre-process our input before feeding it into the model. Hence we define a preprocess function to reshape the images to (299 x 299) and feed to the preprocess_input() function of Keras."""


        def preprocess(image_path):
            img = image.load_img(image_path, target_size=(299, 299))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            return x


        """Now we can go ahead and encode our training and testing images, i.e extract the images vectors of shape (2048,)"""


        def encode(image):
            image = preprocess(image)
            fea_vec = model_new.predict(image)
            fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
            return fea_vec


        self.encoding_train = {}
        cpt = 0

        for img in preprocessedObj.train_img:
            print("image : {}, cpt : {}".format(img, cpt))
            self.encoding_train[img[len(FLICKR_PATH_DIR):]] = encode(img)
            cpt += 1
        train_features = self.encoding_train

        self.encoding_test = {}
        cpt = 0
        for img in preprocessedObj.test_img:
            print("image : {}, cpt : {}".format(img, cpt))
            self.encoding_test[img[len(FLICKR_PATH_DIR):]] = encode(img)
            cpt += 1

        """Save the training and validation into files."""

        now = datetime.now()

        # dd-mm-YY_H-M-S
        dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
        filename_encoding_train = FLICKR_PATH_DIR_8K_5C_METHOD2 + 'encoding_train' + dt_string + '.pkl'
        filename_encoding_test = FLICKR_PATH_DIR_8K_5C_METHOD2 + 'encoding_test' + dt_string + '.pkl'

        # Save the model .pkl file
        dump(self.encoding_train, open(filename_encoding_train, 'wb'))
        dump(self.encoding_test, open(filename_encoding_test, 'wb'))



        # Feature extractor model
        input_1     = Input(shape=(2048,))
        droplayer   = Dropout(0.5)(input_1)
        denselayer  = Dense(256, activation='relu')(droplayer)

        # Sequence model
        input_2     = Input(shape=(preprocessedObj.max_length,))
        embedding   = Embedding(preprocessedObj.vocab_size, preprocessedObj.embedding_dim, mask_zero=True)(input_2)
        droplayer_  = Dropout(0.5)(embedding)
        lstm        = LSTM(256)(droplayer_)

        # Decoder model
        decoder1 = add([denselayer, lstm])
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(preprocessedObj.vocab_size, activation='softmax')(decoder2)

        # Optimizer
        self.optimizer   = Adam()

        # Tie it together [image, seq] [word]
        self.model = Model(inputs=[input_1, input_2],
                      outputs=outputs)

        self.model.summary()

        fn_model = FLICKR_PATH_DIR_8K_5C_METHOD2_MODEL + "model.png"
        plot_model(self.model, to_file=fn_model, show_shapes=True)

        """Input_7 is the partial caption of max length 80 which is fed into the embedding layer. This is where the words are mapped to the 200-d Glove embedding. It is followed by a dropout of 0.5 to avoid overfitting. This is then fed into the LSTM for processing the sequence.
        
        Input_6 is the image vector extracted by our InceptionV3 network. It is followed by a dropout of 0.5 to avoid overfitting and then fed into a Fully Connected layer.
        
        Both the Image model and the Language model are then concatenated by adding and fed into another Fully Connected layer. The layer is a softmax layer that provides probabilities to our 2101 word vocabulary.
        
        # 5 - Model Training
        
        Before training the model we need to keep in mind that we do not want to retrain the weights in our embedding layer (pre-trained Glove vectors).
        """
    def train(self):
        self.model.layers[2].set_weights([self.preprocessedObj.embedding_matrix])
        self.model.layers[2].trainable = False

        """Next, compile the model using Categorical_Crossentropy as the Loss function and Adam as the optimizer."""

        self.model.compile(loss      = 'categorical_crossentropy',
                      optimizer = self.optimizer,
                      metrics   = ["accuracy"])

        """Since our dataset has 6000 images and 40000 captions we will create a function that can train the data in batches."""

        def data_generator(descriptions, photos, wordtoix, max_length, num_photos_per_batch):
            X1, X2, y = list(), list(), list()
            n=0
            # loop for ever over images
            while 1:
                for key, desc_list in descriptions.items():
                    n+=1
                    # retrieve the photo feature
                    photo = photos[key+'.jpg']
                    for desc in desc_list:
                        # encode the sequence
                        seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                        # split one sequence into multiple X, y pairs
                        for i in range(1, len(seq)):
                            # split into input and output pair
                            in_seq, out_seq = seq[:i], seq[i]
                            # pad input sequence
                            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                            # encode output sequence
                            out_seq = to_categorical([out_seq], num_classes=self.preprocessedObj.vocab_size)[0]
                            # store
                            X1.append(photo)
                            X2.append(in_seq)
                            y.append(out_seq)

                    if n==num_photos_per_batch:
                        yield ([np.array(X1), np.array(X2)], np.array(y))
                        X1, X2, y = list(), list(), list()
                        n=0

        print(self.model.metrics)

        """Next, let’s train our model for 30 epochs with batch size of 3 and 2000 steps per epoch.
        
        If we have already trained our model,load the latest checkpoints.
        """

        checkpoint_path = FLICKR_PATH_DIR_8K_5C_METHOD2_MODEL_CHECKPOINTS_DIR + "first_training/" + "model-ep30.ckpt"

        # Since my model has already been trained, we lead the weight using the checkpoints
        self.model.load_weights(checkpoint_path)

        epochs = 30
        batch_size = 3
        steps = len(self.preprocessedObj.train_descriptions)//batch_size

        # Define checkpoint callback
        fn_model_sentence_generation = FLICKR_PATH_DIR_8K_5C_METHOD2_MODEL_CHECKPOINTS_DIR + "second_training/" + 'model-ep{epoch:02d}.ckpt'

        model_checkpoint_callback = ModelCheckpoint(filepath          = fn_model_sentence_generation,
                                                    monitor           = 'accuracy',
                                                    mode              = 'max',
                                                    save_best_only    = True,
                                                    save_weights_only = True)

        self.generator = data_generator(self.preprocessedObj.train_descriptions, self.train_features, self.preprocessedObj.wordtoix, self.preprocessedObj.max_length, batch_size)

        self.model.fit(self.generator,
                  steps_per_epoch = steps,
                  verbose         = 1,
                  epochs          = epochs,
                  callbacks       = [model_checkpoint_callback])

        """Let's save our model. The `.h5f5` format can be loaded without knowing the model architecture as opposed to the `.ckpt`."""

        filepath_model = FLICKR_PATH_DIR_8K_5C_METHOD2_MODEL_CHECKPOINTS_DIR + "second_training/"  + "latest_model_epoch30.h5"
        filepath_history = FLICKR_PATH_DIR_8K_5C_METHOD2_MODEL_CHECKPOINTS_DIR + "second_training/"  + "latest_model_history.pkl"
        # Saving the model with last parameter
        self.model.save(filepath_model)

        # Save history of the model
        dump(self.model.history.history, open(filepath_history, 'wb'))

        """Let's plot our `model.history.history` that contains the informations about `loss` and `accuracy`."""

        # Plot training loss and accuracy
        plt.plot(self.model.history.history['loss'])
        plt.plot(self.model.history.history['accuracy'])
        plt.title('Model training loss and accuracy')
        plt.ylabel('loss/accuarcy')
        plt.xlabel('epochs')
        plt.legend(['training loss', 'training accuracy'], loc='upper right')

        # Save the figure of model training loss and accuracy
        filename_model_history_loss_acc = FLICKR_PATH_DIR_8K_5C_METHOD2_MODEL_CHECKPOINTS_DIR + "second_training/" + "latest_model_history_loss_acc.jpg"
        plt.savefig(filename_model_history_loss_acc)

        # Show the rigure
        plt.show()

        # Plot training loss only
        plt.plot(self.model.history.history['loss'])
        plt.title('Model training loss')
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.legend(['training loss'], loc='upper right')

        # Save the figure of model training loss and accuracy
        filename_model_history_loss = FLICKR_PATH_DIR_8K_5C_METHOD2_MODEL_CHECKPOINTS_DIR + "second_training/" + "latest_model_history_loss.jpg"
        plt.savefig(filename_model_history_loss)

        # Show the rigure
        plt.show()

        # Plot training accuracy only
        plt.plot(self.model.history.history['accuracy'])
        plt.title('Model training accuracy')
        plt.ylabel('accuarcy')
        plt.xlabel('epochs')
        plt.legend(['training accuracy'], loc='upper right')

        # Save the figure of model training loss and accuracy
        filename_model_history_acc = FLICKR_PATH_DIR_8K_5C_METHOD2_MODEL_CHECKPOINTS_DIR + "second_training/" + "latest_model_history_acc.jpg"
        plt.savefig(filename_model_history_acc)

        # Show the figure
        plt.show()
