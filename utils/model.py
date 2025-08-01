import joblib
import numpy as np

from tensorflow.keras.applications import VGG16

def load_model():
    rf_model = joblib.load('./models/rf_base-mae:0.96_r2:0.75.pkl')
    xgb_model = joblib.load('./models/xgb_best-mae:0.72_r2:0.86.pkl')
    lgbm_model = joblib.load('./models/lgbm_best-mae:1.02_r2:0.71.pkl')
    
    return rf_model, xgb_model, lgbm_model

def load_vgg16():
    vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    return vgg16_model

def choquet_integral(weights, inputs):
    sorted_inputs = sorted(enumerate(inputs), key=lambda x: x[1]) 
    indices, sorted_inputs = zip(*sorted_inputs)
    cumulative_weights = np.cumsum([weights[i] for i in indices])
    return np.dot(sorted_inputs, cumulative_weights)