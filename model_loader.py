from keras.models import model_from_json, load_model
import numpy as np
import tensorflow as tf

class ThyroidModel(object):
    def __init__(self, model_json_file, model_weights_file):
        # Load architecture
        try:
            with open(model_json_file, "r") as json_file:
                loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)
            # Load weights
            self.loaded_model.load_weights(model_weights_file)
        except:
            # Fallback for H5 only
            self.loaded_model = load_model(model_weights_file)
        
    def predict_nodule(self, img):
        self.preds = self.loaded_model.predict(img)
        score = self.preds[0][0]
        # Assuming 1 = Malignant, 0 = Benign based on sigmoid output
        if score > 0.5:
            return "Malignant", self.preds
        else:
            return "Benign", self.preds
