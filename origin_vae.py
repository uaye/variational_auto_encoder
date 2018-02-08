import numpy as np
import tensorflow as tf


#one realization of the original VAE


class VAE():
	def __init__(self, input_dim, layer_dims, z_dim, activations, epoch, batch_size,lr):
		self.batch_size=batch_size
		self.layer_dims=layer_dims
		self.input_dim=input_dim
		self.epoch=epoch
		self.lr=lr
		self.z_dim=z_dim
		self.activations=activations
		self.encoder_weight={}
		self.decoder_weight={}
		self.z_parameter={}
		
	
	def activate(self, linear, name):
		if name == 'sigmoid':
			return tf.nn.sigmoid(linear, name='encoded')
		elif name == 'softmax':
			return tf.nn.softmax(linear, name='encoded')
		elif name == 'linear':
			return linear
		elif name == 'tanh':
			return tf.nn.tanh(linear, name='encoded')
		elif name == 'relu':
			return tf.nn.relu(linear, name='encoded')
	
	
	def train(self, X):
	#set all the parameters
		layer_dims=self.layer_dims
		input_dim=self.input_dim
		activations=self.activations
		batch_size=self.batch_size
		z_dim=self.z_dim
		epoch=self.epoch
		lr=self.lr
	#start build the graph
	#define the input layer
		data=tf.placeholder(tf.float64, [None,input_dim])
		with tf.variable_scope("encoder"):
			encoder={
			'W1': tf.get_variable('W1', [input_dim, layer_dims[0]],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float64),
			'b1': tf.get_variable('b1', [layer_dims[0]], initializer=tf.zeros_initializer,dtype=tf.float64),
			'W2': tf.get_variable('W2', [layer_dims[0], layer_dims[1]],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float64),
			'b2': tf.get_variable('b2', [layer_dims[1]], initializer=tf.zeros_initializer,dtype=tf.float64)
			}
		tem1=tf.matmul(data, encoder['W1'])+encoder['b1']
		h1=self.activate(tem1, name=activations[0])
		tem2=tf.matmul(h1, encoder['W2'])+encoder['b2']
		h2=self.activate(tem2, name=activations[1])

		with tf.variable_scope("z_generation"):
			z_parameter={
			'W_z_mean': tf.get_variable('W_z_mean',[layer_dims[1],z_dim], initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float64),
			'b_z_mean': tf.get_variable('b_z_mean', [z_dim], initializer=tf.zeros_initializer,dtype=tf.float64),
			'W_log_z_sigma':tf.get_variable('W_log_z_sigma',[layer_dims[1],z_dim], initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float64),
			'b_log_z_sigma': tf.get_variable('b_log_z_sigma', [z_dim], initializer=tf.zeros_initializer,dtype=tf.float64)}
		
		tem1=tf.matmul(h2, z_parameter['W_z_mean'])+z_parameter['b_z_mean']
		tem2=tf.matmul(h2, z_parameter['W_log_z_sigma'])+z_parameter['b_log_z_sigma']
		z_mean=self.activate(tem1, name=activations[2])
		log_z_sigma=self.activate(tem2, name=activations[2])
		epsilon=tf.random_normal((batch_size, z_dim), dtype=tf.float64)
		z=epsilon*tf.sqrt(tf.maximum(tf.exp(log_z_sigma),1e-10))+z_mean
		
		with tf.variable_scope("decoder"):
			decoder={
			'W1': tf.get_variable('W1', [z_dim,layer_dims[1]],dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer()),
			'b1': tf.get_variable('b1', [layer_dims[1]], initializer=tf.zeros_initializer,dtype=tf.float64),
			'W2': tf.get_variable('W2', [layer_dims[1], batch_size],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float64),
			'b2': tf.get_variable('b2', [batch_size], initializer=tf.zeros_initializer,dtype=tf.float64),
			'W3': tf.get_variable('W3', [batch_size, input_dim],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float64),
			'b3': tf.get_variable('b3', [input_dim], initializer=tf.zeros_initializer,dtype=tf.float64)
			}			
		tem1=tf.matmul(z, decoder['W1'])+decoder['b1']
		h1=self.activate(tem1, name=activations[2])
		tem2=tf.matmul(h1, decoder['W2'])+decoder['b2']
		h2=self.activate(tem2, name=activations[1])
		tem3=tf.matmul(h2, decoder['W3'])+decoder['b3']
		x_new=self.activate(tem3, name=activations[0])
	#loss function
		gen_loss = -tf.reduce_mean(tf.reduce_sum(data * tf.log(tf.maximum(x_new, 1e-10))+ (1-data) * tf.log(tf.maximum(1 - x_new, 1e-10)),1))
		latent_loss = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(z_mean) + tf.exp(log_z_sigma)- log_z_sigma - 1, 1))
		loss=gen_loss+latent_loss
	#training options:
		train_ops=tf.train.AdamOptimizer(lr).minimize(loss)
		
	#start training the network:
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for i in range(epoch):
				print(sess.run((loss, gen_loss, latent_loss),feed_dict={data:X}))
			


			
np.random.seed(0)

A=np.random.randn(100).reshape(10,10)		
test=VAE(input_dim=10, layer_dims=[5,5,5], z_dim=5, activations=['sigmoid']*3, epoch=10, batch_size=10, lr=0.001)
test.train(A)
