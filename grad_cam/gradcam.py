# import the necessary packages
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model, class_idx, layer_name=None):
        """
		store the model, the class index used to measure the class
		activation map, and the layer to be used when visualizing
		the class activation map
		"""
        self.model = model
        self.classIdx = class_idx
        self.layerName = layer_name

        if self.layerName is None:
            self.layerName = self.find_target_layer()

        self._gradient_model = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, self.model.output])

    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name

        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def __call__(self, image, epsilon=1e-8):
        """
		construct our gradient model by supplying
		(1) the inputs to our pre-trained model,
		(2) the output of the (presumably) final 4D layer in the network,
		(3) the output of the softmax activations from the model
		"""

        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (conv_outputs, predictions) = self._gradient_model(inputs)
            loss = predictions[:, self.classIdx]

        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, conv_outputs)

        # compute the guided gradients
        cast_conv_outputs = tf.cast(conv_outputs > 0, "float32")
        cast_grads = tf.cast(grads > 0, "float32")
        guided_grads = cast_conv_outputs * cast_grads * grads

        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        conv_outputs, guided_grads = conv_outputs[0], guided_grads[0]

        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)

        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        heatmap = self.normalize_heatmap(heatmap, epsilon=epsilon)

        return heatmap

    @staticmethod
    def draw_heatmap_on_image(heatmap, image, alpha=0.5,
                              colormap=cv2.COLORMAP_RAINBOW):
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

        return (heatmap, output)

    @staticmethod
    def normalize_heatmap(heatmap, epsilon):
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + epsilon
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        return heatmap
