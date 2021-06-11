from method2 import FLICKR_SEED_PATH_DIR ,FLICKR_PATH_CSV, FILEPATH_MODEL,FLICKR_PATH_DIR, FLICKR_SEED_PATH_DIR, FLICKR_PATH_DIR_8K_5C_METHOD2, FLICKR_PATH_DIR_8K_5C_METHOD2_MODEL  ,FLICKR_PATH_DIR_8K_5C_METHOD2_MODEL_CHECKPOINTS_DIR, GLOVE_PATH
import matplotlib.pyplot as plt
from IPython.display import Image
# Display one image
from search_utils import SearchUtils
from keras.models import load_model

class Tests:
    def runTests(self,trainedModel, preprocessingObj):
        Image(FLICKR_PATH_DIR + preprocessingObj.flickr_csv["image_name"][0])
        # Display the first comment (caption) of the above image
        preprocessingObj.flickr_csv["comment"][0]

        image_name = preprocessingObj.flickr_csv["image_name"][30000]
        self.runTest(image_name,trainedModel,preprocessingObj)
        image_name = preprocessingObj.flickr_csv["image_name"][30005]
        self.runTest(image_name,trainedModel,preprocessingObj)

        image = trainedModel.encoding_test[image_name].reshape((1,2048))
        self.runTest(image_name,trainedModel,preprocessingObj)
        image_name = preprocessingObj.flickr_csv["image_name"][0]
        self.runTest(image_name, trainedModel, preprocessingObj)


        model = load_model(FILEPATH_MODEL)
        model.summary()


    def runTest(self,trainedModel, preprocessingObj, image_name):
        image = trainedModel.encoding_test[image_name].reshape((1, 2048))
        x = plt.imread(FLICKR_PATH_DIR + image_name)
        plt.imshow(x)
        plt.show()

        print("Greedy Search:",
              SearchUtils.greedySearch(image, trainedModel, preprocessingObj.max_length, preprocessingObj.wordtoix,preprocessingObj.ixtoword))
        print("Beam Search, K = 3:",
              SearchUtils.beam_search_predictions(image, trainedModel.model, preprocessingObj.max_length,
                                                  preprocessingObj.wordtoix, preprocessingObj.ixtoword, beam_index=3))
        print("Beam Search, K = 5:",
              SearchUtils.beam_search_predictions(image, trainedModel.model, preprocessingObj.max_length,
                                                  preprocessingObj.wordtoix, preprocessingObj.ixtoword, beam_index=5))
        print("Beam Search, K = 7:",
              SearchUtils.beam_search_predictions(image, trainedModel.model, preprocessingObj.max_length,
                                                  preprocessingObj.wordtoix, preprocessingObj.ixtoword, beam_index=7))
        print("Beam Search, K = 10:",
              SearchUtils.beam_search_predictions(image, trainedModel.model, preprocessingObj.max_length,
                                                  preprocessingObj.wordtoix, preprocessingObj.ixtoword, beam_index=10))


