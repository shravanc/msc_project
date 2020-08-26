import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf




check_point_path = '/home/shravan/dissertation/bert_model/saved_model/1'
model = tf.keras.models.load_model(check_point_path)


print(model.summary())
#image = '/home/shravan/aws_scripts/cronjob/model.png'
#tf.keras.utils.plot_model(model, to_file=image, show_shapes=True, show_layer_names=True)
#plt.show()
