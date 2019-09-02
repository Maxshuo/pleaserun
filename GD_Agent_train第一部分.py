import numpy as np
import tflearn
from tflearn.layers.estimator import regression
import tensorflow as tf

path = 'data_set_OGD.npz'
res = np.load(path)
data_set = res['arr_0']
train_input = np.array([sample[0] for sample in data_set])
train_output = np.array([sample[1] for sample in data_set])

X = train_input[:350000,:]
Y = train_output[:350000,:]

testX = train_input[350000:,:]
testY = train_output[350000:,:]

Layer1 = 400
Layer2 = 300

inputs = tflearn.input_data(shape=[None, 3], name="input_1")
net = tflearn.fully_connected(inputs, Layer1)
net = tflearn.layers.normalization.batch_normalization(net)
net = tflearn.activations.relu(net)
net = tflearn.fully_connected(net, Layer2)
net = tflearn.layers.normalization.batch_normalization(net)
net = tflearn.activations.relu(net)
# Final layer weights are init to Uniform[-3e-3, 3e-3]
w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
out = tflearn.fully_connected(net, 2, activation='sigmoid', weights_init=w_init)
# Scale output to -action_bound to action_bound
scaled_out = tf.multiply(out, 2.0, name="output_1")

network = regression(scaled_out, optimizer='adam', learning_rate=0.0001,
                     loss='mean_square', name='target')

model = tflearn.DNN(network, tensorboard_verbose=0)
# pdb.set_trace()
model.fit({'input_1': X}, {'target': Y}, n_epoch=5,
           validation_set=({'input_1': testX}, {'target': testY}),
           snapshot_step=100, show_metric=True, run_id='convnet_mnist')
model.predict(testX[100:110,:]),testY[100:110,:]
