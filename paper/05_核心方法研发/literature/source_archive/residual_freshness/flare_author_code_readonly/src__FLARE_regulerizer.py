
from tensorflow.keras import regularizers
import tensorflow as tf
import tensorflow_probability as tfp

@tf.keras.utils.register_keras_serializable(package='Custom', name='FLARE_REGULARIZATION')
class FLARE_REGULARIZATION(regularizers.Regularizer):
    def __init__(self, tau,t, accumolator_tensor,server_weights_tensor):
        self.tau = tau
        self.t = t
        self.accumolator_tensor = accumolator_tensor
        self.server_weights = server_weights_tensor
    def __call__(self, w):       
        #treshhold = tf.math.reduce_mean(tf.math.abs(self.accumolator_tensor))
        treshhold = tfp.stats.percentile(tf.math.abs(self.accumolator_tensor), q=self.t)
        mask =  tf.cast(tf.greater(tf.math.abs(self.accumolator_tensor), treshhold), tf.float32) #those who are the smallest are most updated in acc
        return (self.tau) * tf.reduce_sum(tf.math.multiply(mask,tf.math.pow(tf.math.abs(tf.subtract( w,tf.add(self.accumolator_tensor,self.server_weights))),1))) 
    
    def get_config(self):
        return {'tau': self.tau,
                't': self.t,
                'accumolator': self.accumolator_tensor,
                'server_weights': self.server_weights }

