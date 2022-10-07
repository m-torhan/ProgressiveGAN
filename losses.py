import tensorflow.keras.backend as K
import numpy as np

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def gradient_penalty_loss(y_true, y_pred, interpolated_samples):
    gradients = K.gradients(y_pred, interpolated_samples)[0]
    
    gradient_l2_norm = K.sqrt(K.sum(K.square(gradients),  axis=np.arange(1, len(gradients.shape))))
    
    gradient_penalty = K.square(1 - gradient_l2_norm)
    
    return K.mean(gradient_penalty)