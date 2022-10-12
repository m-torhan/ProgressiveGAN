import tensorflow.keras.backend as K

from keras.layers import Layer, Add

class WeightedSum(Add):
	# init with default value
	def __init__(self, alpha=0.0, **kwargs):
		super(WeightedSum, self).__init__(**kwargs)
		self.alpha = K.variable(alpha, name='ws_alpha')
 
	# output a weighted sum of inputs
	def _merge_function(self, inputs):
		output = (self.alpha*inputs[0]) + ((1.0 - self.alpha)*inputs[1])
		return output

class PixelNormalization(Layer):
    '''
    pixel-wise feature vector normalization layer
    '''
    def __init__(self, **kwargs):
        super(PixelNormalization, self).__init__(**kwargs)
 
    def call(self, inputs):
        values = inputs**2.0
        mean_values = K.mean(values, axis=-1, keepdims=True)
        mean_values += 1.0e-8
        l2 = K.sqrt(mean_values)
        normalized = inputs / l2
        return normalized
 
    def compute_output_shape(self, input_shape):
        return input_shape
    
class MinibatchStdev(Layer):
    '''
    mean standard deviation across each pixel coord
    '''
    def __init__(self, **kwargs):
        super(MinibatchStdev, self).__init__(**kwargs)

    def call(self, inputs):
        mean = K.mean(inputs, axis=0, keepdims=True)
        mean_sq_diff = K.mean(K.square(inputs - mean), axis=0, keepdims=True) + 1e-8
        mean_pix = K.mean(K.sqrt(mean_sq_diff), keepdims=True)
        shape = K.shape(inputs)
        output = K.tile(mean_pix, [shape[0], shape[1], shape[2], 1])
        return K.concatenate([inputs, output], axis=-1)

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        input_shape[-1] += 1
        return tuple(input_shape)
    
class RandomWeightedAverage(Layer):
    def __init__(self, **kwargs):
        super(RandomWeightedAverage, self).__init__(**kwargs)
    
    def call(self, inputs):
        alpha = K.random_uniform(K.shape(inputs[0])[:1])
        alpha = K.reshape(alpha, (-1, 1, 1, 1))
        return (alpha*inputs[0]) + ((1 - alpha)*inputs[1])
    
    def compute_output_shape(self, input_shape):
        return input_shape[0]