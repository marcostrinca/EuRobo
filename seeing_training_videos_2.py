import tensorflow as tf

########## carregando as imagens geradas pela webcam em numpy array para ser input da CNN
from PIL import Image
import sys, os
import numpy as np
from scipy import ndimage
from scipy.misc import imresize

########## funçãozinha auxiliar para plotar imagens de teste
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

def plot_image(image):
	plt.imshow(image, cmap="gray", interpolation="nearest")
	plt.axis("off")
########## ----------

########## TRAINING DATA
labels = np.array([name for name in os.listdir("./Data/gestos")])
n_labels = len(labels)
print(labels)

# frame specs
n_frames = 120
n_total_frames = n_frames * n_labels

img_width = 128
img_height = 96
size = img_width, img_height
n_channels = 1
pixel_depth = 255

# prepare data
idx_x = 0
idx_folder = 0
X_arr = np.zeros((n_total_frames, 96, 128))
Y_arr = np.zeros((n_total_frames,))

for folder in labels:

	# path das pastas
	path = 'Data/gestos/{0}'.format(folder)
	print(path)

	# pra cada arquivo na pasta 
	for filename in os.listdir(path):
		
		# abro a imagem
		img = Image.open(os.path.join(path, filename))

		# resize pro tamanho final esperado
		img.thumbnail(size, Image.ANTIALIAS)

		# converto para numpy array
		img = np.asarray(img)

		# normalizo os valores da imagem 
		image_data = (img.astype(float) - pixel_depth / 2) / pixel_depth
		
		X_arr[idx_x] = image_data
		Y_arr[idx_x,] = idx_folder

		idx_x += 1
	idx_folder += 1

print(X_arr[0][0])
print('Mean: ', np.mean(X_arr))
print('Standar Deviation: ', np.std(X_arr))

# split and shuffle the data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_arr, Y_arr, test_size=0.2, random_state=42)

print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print('X_test shape:', X_test.shape)
print('Y_test shape:', Y_test.shape)

# reshape for tensorflow
def reformat(dataset, labels):
	dataset = dataset.reshape((-1, img_height, img_width, n_channels)).astype(np.float32)
	labels = (np.arange(n_labels) == labels[:,None]).astype(np.float32)
	return dataset, labels

train_dataset, train_labels = reformat(X_train, Y_train)
test_dataset, test_labels = reformat(X_test, Y_test)
print('Training set', train_dataset.shape, train_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
print(test_labels[146])

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

batch_size = 80

c1_depth = 6
c1_ker_sz = 5
c3_depth = 32
c3_ker_sz = 6
c5_depth = 120
c5_ker_sz = 6

num_hidden = 512

graph = tf.Graph()

with graph.as_default():

	# Input data.
	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, img_height, img_width, n_channels))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, n_labels))
	tf_test_dataset = tf.constant(test_dataset)

	c1_weights = tf.Variable(tf.truncated_normal([c1_ker_sz, c1_ker_sz, n_channels, c1_depth], stddev=0.1))
	c1_biases = tf.Variable(tf.zeros([c1_depth]))

	c3_weights = tf.Variable(tf.truncated_normal([c3_ker_sz, c3_ker_sz, c1_depth, c3_depth], stddev=0.1))
	c3_biases = tf.Variable(tf.constant(1.0, shape=[c3_depth]))

	c5_weights = tf.Variable(tf.truncated_normal([c5_ker_sz, c5_ker_sz, c3_depth, c5_depth], stddev=0.1))
	c5_biases = tf.Variable(tf.constant(1.0, shape=[c5_depth]))
  
	c5_conv_dim_h = (((((img_height+1)//2) + 1) // 2) + 1 )//2
	c5_conv_dim_w = (((((img_width+1)//2) + 1) // 2) + 1 )//2

	fc_weights = tf.Variable(tf.truncated_normal([c5_conv_dim_h * c5_conv_dim_w * c5_depth, num_hidden], stddev=0.1))
	fc_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
  
	out_weights = tf.Variable(tf.truncated_normal([num_hidden, n_labels], stddev=0.1))
	out_biases = tf.Variable(tf.constant(1.0, shape=[n_labels]))
  
	# Model.
	def model(data):
		print(data.get_shape().as_list())

		conv = tf.nn.conv2d(data, c1_weights, [1, 1, 1, 1], padding='SAME')
		hidden = tf.nn.relu(conv + c1_biases)
		print(conv.get_shape().as_list())

		pooled = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
		print(pooled.get_shape().as_list())

		conv = tf.nn.conv2d(pooled, c3_weights, [1, 1, 1, 1], padding='SAME')
		hidden = tf.nn.relu(conv + c3_biases)
		pooled = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
		shape = pooled.get_shape().as_list()
		print(shape)

		conv = tf.nn.conv2d(pooled, c5_weights, [1, 1, 1, 1], padding='SAME')
		hidden = tf.nn.relu(conv + c5_biases)
		pooled = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
		shape = pooled.get_shape().as_list()
		print(shape)

		reshape = tf.reshape(pooled, [shape[0], shape[1] * shape[2] * shape[3]])
		hidden = tf.nn.relu(tf.matmul(reshape, fc_weights) + fc_biases)

		return tf.matmul(hidden, out_weights) + out_biases
  
	# Training computation.
	logits = model(tf_train_dataset)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
	
	# Optimizer.
	# optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
	optimizer = tf.train.AdagradOptimizer(0.001).minimize(loss)

	# batch = tf.Variable(0)
	# learning_rate = tf.train.exponential_decay(
	#    0.01,                # Base learning rate.
	#    batch * batch_size,  # Current index into the dataset.
	#    train_labels.shape[0],          # Decay step.
	#    0.95,                # Decay rate.
	#    staircase=True)
	# optimizer = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss, global_step=batch)
	  
	# Predictions for the training, validation, and test data.
	train_prediction = tf.nn.softmax(logits)
	test_prediction = tf.nn.softmax(model(tf_test_dataset))

	saver = tf.train.Saver()

num_steps = 1000
with tf.Session(graph=graph) as session:
	tf.global_variables_initializer().run()
	print('Initialized')

	for step in range(num_steps):
		offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
		batch_data = train_dataset[offset:(offset + batch_size), :]
		batch_labels = train_labels[offset:(offset + batch_size), :]
		feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}

		_, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

		if (step % 20 == 0):
			print('Minibatch loss at step %d: %f' % (step, l))
			print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
			print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

	save_path = saver.save(session, "Data/checkpoints/model_gestos2.ckpt")
