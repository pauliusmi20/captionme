# GLOBAL VARIABLES
# Paths to the dataset
# Path to data folderZ

##Variables "env"
FLICKR_SEED_PATH_DIR = "../flickr30k_images/"
#Path to comments for images
FLICKR_PATH_CSV = FLICKR_SEED_PATH_DIR + "results.csv"
#Path to image dir
FLICKR_PATH_DIR = FLICKR_SEED_PATH_DIR + "flickr30k_images/"
# Path to 5C method
FLICKR_PATH_DIR_8K_5C_METHOD2 = FLICKR_SEED_PATH_DIR + "flickr8K_5C_Method2/"
#Path model export
FLICKR_PATH_DIR_8K_5C_METHOD2_MODEL = FLICKR_PATH_DIR_8K_5C_METHOD2 + "model/"
#Path model checkpoint export
FLICKR_PATH_DIR_8K_5C_METHOD2_MODEL_CHECKPOINTS_DIR = FLICKR_PATH_DIR_8K_5C_METHOD2_MODEL + "model_generation_sentences/"
#Path glove
GLOVE_PATH = FLICKR_PATH_DIR_8K_5C_METHOD2 + 'glove6b/'
#Model filepath second training
FILEPATH_MODEL = FLICKR_PATH_DIR_8K_5C_METHOD2_MODEL_CHECKPOINTS_DIR + "second_training/" + "latest_model_epoch30.h5"

from method2.model import Model
from method2.preprocessing import PreprocessingM2
from method2.test import Tests

if __name__ == "__main__":
    #Preprocessing data
    precprocessedObj = PreprocessingM2()
    model = Model(precprocessedObj)
    # TRAIN the model
    model.train()
    # Run tests
    Tests.runTests(model,precprocessedObj)


