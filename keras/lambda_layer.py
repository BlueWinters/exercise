import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation,Reshape
from keras.layers.merge import Concatenate
from keras.utils.vis_utils import plot_model
from keras.layers import Input, Lambda
from keras.models import Model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def slice(x,index):
    return x[:,:,index]

a = Input(shape=(4,2))
x1 = Lambda(slice,output_shape=(4,1),arguments={'index':0})(a)
x2 = Lambda(slice,output_shape=(4,1),arguments={'index':1})(a)
x1 = Reshape((4,1,1))(x1)
x2 = Reshape((4,1,1))(x2)
# output = merge([x1,x2], mode='concat')
output = Concatenate(axis=2)([x1, x2])
model = Model(a, output)
x_test = np.array([[[1,2],[2,3],[3,4],[4,5]]])
print(model.predict(x_test))
# plot_model(model, to_file='lambda.png', show_shapes=True)
