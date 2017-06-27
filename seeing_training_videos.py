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

# crop image by the center
def crop_center(img,cropx,cropy):
	y,x = img.shape
	startx = x//2-(cropx//2)
	starty = y//2-(cropy//2)    
	return img[starty:starty+cropy,startx:startx+cropx]

########## ----------

########## TRAINING DATA
labels = np.array([name for name in os.listdir("./Data/gestos")])
print(labels.shape)

n_labels = len(labels)
n_frames = 60
n_total_frames = n_frames * n_labels
size = 128, 96

idx_x = 0
idx_folder = 0
X_arr = np.zeros((n_total_frames,96,128))
Y_arr = np.zeros((n_total_frames,))
for folder in labels:

	# path das pastas
	path = 'Data/gestos/{0}'.format(folder)
	print(path)
	
	# bora ler os arquivos
	# cria uma funcaozinha para ler os arquivos
	read = lambda imname: Image.open(imname)

	# pra cada arquivo na pasta 
	for filename in os.listdir(path):
		
		# adiciono ao array no index idx_x a imagem e o label (que e o numero da pasta)
		img = Image.open(os.path.join(path, filename))
		img.thumbnail(size, Image.ANTIALIAS)

		img = np.asarray(img)
		# img = crop_center(img, 128, 96)
		
		X_arr[idx_x,] = img
		Y_arr[idx_x,] = idx_folder

		idx_x = idx_x + 1

	idx_folder += 1
	

print('X_arr shape:', X_arr.shape)
print('Y_arr shape:', Y_arr.shape)

print('image for label:', Y_arr[60])
plot_image(X_arr[60, :, :])
plt.show()

#### SPLIT TRAINING DATA AND TEST DATA
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_arr, Y_arr, test_size=0.2, random_state=42)

# plot_image(X_train[2, :, :])
# plt.show()
# sys.exit(0)



########## arquitetura da CNN: ResNet-34 - baseado no capítulo 13 do livro hands on machine learning

# width * height = 9216
X_train = np.reshape(X_train, (-1, 12288))
X_test = np.reshape(X_test, (-1, 12288))

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

height = 96
width = 128
channels = 1
n_inputs = height * width

conv1_fmaps = 32
conv1_ksize = 3
conv1_stride = 1
conv1_pad = "SAME"

conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 2
conv2_pad = "SAME"

pool3_fmaps = conv2_fmaps

n_fc1 = 64
n_outputs = n_labels

graph = tf.Graph()
with graph.as_default():
	with tf.name_scope("inputs"):
		X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
		X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
		y = tf.placeholder(tf.int32, shape=[None], name="y")

	conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize, strides=conv1_stride, padding=conv1_pad, activation=tf.nn.relu, name="conv1")
	conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize, strides=conv2_stride, padding=conv2_pad, activation=tf.nn.relu, name="conv2")

	with tf.name_scope("pool3"):
		pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
		pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 24 * 32])

	with tf.name_scope("fc1"):
		fc1 = tf.layers.dense(pool3_flat, n_fc1, activation=tf.nn.relu, name="fc1")

	with tf.name_scope("output"):
		logits = tf.layers.dense(fc1, n_outputs, name="output")
		Y_proba = tf.nn.softmax(logits, name="Y_proba")

	with tf.name_scope("train"):
		xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
		loss = tf.reduce_mean(xentropy)
		optimizer = tf.train.AdamOptimizer()
		training_op = optimizer.minimize(loss)

	with tf.name_scope("eval"):
		correct = tf.nn.in_top_k(logits, y, 1)
		accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

	with tf.name_scope("init_and_save"):
		init = tf.global_variables_initializer()
		saver = tf.train.Saver()


# get the model state
def get_model_params():
	gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
	return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

# restore a previous state
def restore_model_params(model_params):
	gvar_names = list(model_params.keys())
	assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
				  for gvar_name in gvar_names}
	init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
	feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
	tf.get_default_session().run(assign_ops, feed_dict=feed_dict)



## treinando a parada toda
n_epochs = 35
batch_size = 20
best_model_params = None 

with tf.Session(graph=graph) as sess:
	init.run()
	for epoch in range(n_epochs):
		offset = (epoch * batch_size) % (Y_train.shape[0] - batch_size)
		batch_data = X_train[offset:(offset + batch_size), :]
		batch_labels = Y_train[offset:(offset + batch_size)]

		sess.run(training_op, feed_dict={X: batch_data, y: batch_labels})
		acc_train = accuracy.eval(feed_dict={X: X_train, y: Y_train})
		acc_test = accuracy.eval(feed_dict={X: X_test, y: Y_test})
		print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

	if best_model_params:
		restore_model_params(best_model_params)

	save_path = saver.save(sess, "Data/gestos/model_gestos.ckpt")

