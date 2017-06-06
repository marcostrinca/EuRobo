import tensorflow as tf

########## carregando as imagens geradas pela webcam em numpy array para ser input da CNN
from PIL import Image
import sys, os
import numpy as np
from scipy import ndimage


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
emos = 6
idx_folder = 0

idx_x = 0
X_arr = np.zeros((20*emos,48,48))
Y_arr = np.zeros((20*emos,))
while idx_folder < emos:
	# path das pastas
	folder = 'Data/images/{0}'.format(idx_folder)
	
	# bora ler os arquivos
	read = lambda imname: np.asarray(Image.open(imname))

	# pra cada arquivo na pasta 0, depois 1, depois 2, depois 3 etc...
	for filename in os.listdir(folder):
		
		# adiciono ao array no index idx_x a imagem e o label (que e o numero da pasta)
		X_arr[idx_x,] = read(os.path.join(folder, filename))
		Y_arr[idx_x,] = idx_folder

		idx_x = idx_x + 1
	

	# incremento
	idx_folder = idx_folder + 1

print('X_arr shape:', X_arr.shape)
print('qdd de imagens: ',len(X_arr))

#### SPLIT TRAINING DATA AND TEST DATA

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_arr, Y_arr, test_size=0.2, random_state=42)

X_train = np.reshape(X_train, (-1, 2304))
X_test = np.reshape(X_test, (-1, 2304))

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# plot_image(X_train[2, :, :])
# plt.show()

########## arquitetura da CNN: ResNet-34 - baseado no capítulo 13 do livro hands on machine learning
height = 48
width = 48
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

n_fc1 = 128
n_outputs = 6

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
        pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 12 * 12])

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

# help pra fazer o batch
# def next_batch(num, data, labels):
#     '''
#     Return a total of `num` random samples and labels. 
#     '''
#     idx = np.arange(0 , len(data))
#     np.random.shuffle(idx)
#     idx = idx[:num]
#     data_shuffle = [data[ i] for i in idx]
#     labels_shuffle = [labels[ i] for i in idx]

#     return np.asarray(data_shuffle), np.asarray(labels_shuffle)


## treinando a parada toda
n_epochs = 1000
batch_size = 2

best_acc_val = 0
check_interval = 100
checks_since_last_progress = 0
max_checks_without_progress = 100
best_model_params = None 

with tf.Session(graph=graph) as sess:
    init.run()
    for epoch in range(n_epochs):
        sess.run(training_op, feed_dict={X: X_train, y: Y_train})
        acc_train = accuracy.eval(feed_dict={X: X_train, y: Y_train})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: Y_test})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

    if best_model_params:
        restore_model_params(best_model_params)

    save_path = saver.save(sess, "Data/model_seeing.ckpt")



    # for epoch in range(n_epochs):
    #     for iteration in range(len(X_train) // batch_size):
    #         X_batch, y_batch = next_batch(batch_size, X_train, Y_train)
    #         sess.run(training_op, feed_dict={X: X_batch, y: y_batch, is_training: True})
    #         if iteration % check_interval == 0:
    #             acc_val = accuracy.eval(feed_dict={X: X_train, y: Y_train})
    #             if acc_val > best_acc_val:
    #                 best_acc_val = acc_val
    #                 checks_since_last_progress = 0
    #                 best_model_params = get_model_params()
    #             else:
    #                 checks_since_last_progress += 1
    #     acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
    #     acc_test = accuracy.eval(feed_dict={X: X_test, y: Y_test})
    #     print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test, "Best validation accuracy:", best_acc_val)
    #     if checks_since_last_progress > max_checks_without_progress:
    #         print("Early stopping!")
    #         break

    # if best_model_params:
    #     restore_model_params(best_model_params)
    # acc_test = accuracy.eval(feed_dict={X: X_test, y: Y_test})
    # print("Final accuracy on test set:", acc_test)
    # save_path = saver.save(sess, "./my_mnist_model")