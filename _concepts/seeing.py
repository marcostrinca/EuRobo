import argparse, os, sys, time

import numpy as np
from scipy import ndimage

# video  and utils
from er_utils.video import ER_WebcamVideoStream
from er_utils.video import ER_FPS
import face_detection_utilities as fdu
import cv2

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# algumas variáveis necessárias
windowsName = 'Preview Screen'
REC_COLOR = (0, 255, 0)
FACE_SHAPE = (48, 48)

emo = ['Sorridente', 'Triste', 'Surpreso', 'Medo', 'Raiva', 'Neutro', 'Maluco']
gesture_labels = ['aquilo', 'conhecer', 'eu', 'gostar', 'isso', 'legal', 'muito', 'nao_querer', 'oi', 'querer', 'saber', 'voce']

# inputs
parser = argparse.ArgumentParser(description='Módulo de visão do EuRobo')
parser.add_argument('-test', 
	help=('Dado o caminho de uma imagem de teste, o app tenta prever a expressão facial.'))
parser.add_argument('-train', 
	help=('Captura 20 imagens da webcam para serem usadas em treinamento de CNN. Opções: 0-sorridente, 1-triste, 2-surpreso, 3-medo, 4-raiva, 5-normal .'))
parser.add_argument('-train_video_sequence', 
	help=('Captura imagens da webcam durante 3 segundos para serem usadas em treinamento de CNN. Opções: string com o label da sequencia'))
args = parser.parse_args()

########## MAIN FUNCTION
def main():
	print("Inicializando módulo de visão bot")

	# inicio captura
	capture = ER_WebcamVideoStream().start()
	if not capture:
		print("Falha capturando video streaming")
		sys.exit(1)
	else:
		print("Sucesso capturando video streaming")

	# lido com os argumentos
	if args.test is not None:
		img = cv2.imread(args.test)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, FACE_SHAPE)
		print(class_label[result[0]])
		sys.exit(0)

	if args.train is not None:
		getFacesToTrain(capture, args.train, 20)

	if args.train_video_sequence is not None:
		getVideoSequenceToTrain(capture, args.train_video_sequence, 10)
	
	# showScreenAndDectect(capture)
	# while (True):
	#     time.sleep(0.1)
	#     frame = capture.read()
	#     cv2.imshow("Frame", frame)
	#     if cv2.waitKey(1) & 0xFF == ord('q'):
	#         break

	# testScreen(capture)
	testGestures(capture)

def testScreen(capture):
	fps = ER_FPS().start()
	while (True):
		time.sleep(0.011) # target 90fps
		frame = capture.read()

		faceCoordinates = fdu.getFaceCoordinates(frame)
		if faceCoordinates is not None:
			cv2.rectangle(np.asarray(frame), (faceCoordinates[0], faceCoordinates[1]), (faceCoordinates[2], faceCoordinates[3]), REC_COLOR, thickness=2)

		cv2.imshow(windowsName, frame)
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

		fps.update()

	fps.stop()
	print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	capture.stop()
	cv2.destroyAllWindows()

def testGestures(capture):

	fps = ER_FPS().start()
	while (True):
		time.sleep(0.1) # target 90fps
		frame = capture.read()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frame = cv2.resize(frame, (128, 96))
		img = np.asarray(frame)
		img = (img.astype(float) - 255 / 2) / 255

		cv2.imshow(windowsName, frame)
		if cv2.waitKey(1) & 0xFF == ord("q"):
		    break

		img = np.reshape(img, (-1, 96, 128, 1))

		# predict
		test_g = predict_gestures2(img)
		pred_sort = test_g.argsort()
		print(pred_sort)
		print(pred_sort.shape)

		r1 = pred_sort[0,-1]
		r2 = pred_sort[0,-2]
		r3 = pred_sort[0,-3]
		print("option 1: ", gesture_labels[r1])
		print("option 2: ", gesture_labels[r2])
		print("option 3: ", gesture_labels[r3])

		fps.update()

	fps.stop()
	print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	capture.stop()
	cv2.destroyAllWindows()



########## Detectando faces, recortando, tratando e mandando pro algoritmo de ML
def showScreenAndDectect(capture):
	while (True):

		time.sleep(0.1)

		# começa a capturar 
		frame = capture.read()

		# coordenadas do rosto no esquema [0,0,0,0]
		faceCoordinates = fdu.getFaceCoordinates(frame)
		
		if faceCoordinates is not None:
			
			# desenha um retangulo ao redor do rosto
			cv2.rectangle(np.asarray(frame), (faceCoordinates[0], faceCoordinates[1]), (faceCoordinates[2], faceCoordinates[3]), REC_COLOR, thickness=2)

			# crop apenas na regiao da face 
			face = frame[faceCoordinates[1]:faceCoordinates[3], faceCoordinates[0]:faceCoordinates[2]]

			# resize pro tamanho esperado pelo algoritmo
			face_scaled = cv2.resize(face, (48,48))

			# converte pra grayscale
			face_img = cv2.cvtColor(face_scaled, cv2.COLOR_BGR2GRAY)

			# mostra os frames numa janela
			# cv2.startWindowThread()
			cv2.imshow(windowsName, frame)
			
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

			# descobre a expressão facial com base na imagem (converto pro modelo que o tf está esperando)
			# img = np.reshape(face_img, (-1, 2304))

			# test_emo = predict(img)
			# index_emo = test_emo[0]
			# print (emo[index_emo])

		return True

	capture.release()
	cv2.destroyAllWindows()

########## Treinando reconhecimento de faces 
def getFacesToTrain(capture, emo, iteractions):
	count = 0
	while count < iteractions:

		# inicializo captura
		frame = capture.read()

		# coordenadas [0,0,0,0]
		faceCoordinates = fdu.getFaceCoordinates(frame)
		
		if faceCoordinates is not None:
			
			# retangulo ao redor do rosto
			cv2.rectangle(np.asarray(frame), (faceCoordinates[0], faceCoordinates[1]), (faceCoordinates[2], faceCoordinates[3]), REC_COLOR, thickness=2)

			# crop da imagem de acordo com o retangulo
			face = frame[faceCoordinates[1]:faceCoordinates[3], faceCoordinates[0]:faceCoordinates[2]]

			# resize para o tamanho esperado
			face_scaled = cv2.resize(face, (48,48))

			# grayscale
			face_img = cv2.cvtColor(face_scaled, cv2.COLOR_BGR2GRAY)

			# salvo as imagens no disco
			print("Salvando imagem to:", 'Data/images/{0}/img_{1}.jpg'.format(emo, count))
			cv2.imwrite('data/images/{0}/img_{1}.jpg'.format(emo, count), face_img)

			count = count + 1
			time.sleep(0.33)
		print(count)


	print("Pronto. Capturadas as imagens para a expressão:", emo)
	sys.exit(0)
##########################################

########## Treinando gestos 
def getVideoSequenceToTrain(capture, label, secs):

	# checo se a pasta com o label ja existe

	# inicializo o timer e contador de imagens
	count = 0
	start_timer = time.time()
	while (True):

		time.sleep(0.033) # target fps

		# inicializo captura
		frame = capture.read()

		# resize para o tamanho esperado
		# frame_scaled = cv2.resize(frame, (128,128))

		# grayscale
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# salvo as imagens no disco
		print("Salvando imagem to:", 'Data/gestos/{0}/img_{1}.jpg'.format(label, count))
		cv2.imwrite('Data/gestos/{0}/img_{1}.jpg'.format(label, count), frame)

		count = count + 1

		print(count)
		if (time.time() > start_timer + secs) or count == 120:
			break

	print("Pronto. Capturadas as imagens para o label:", label)
	capture.stop()
	cv2.destroyAllWindows()

##########################################
def predict_faces(img):
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

	# predict
	with tf.Session(graph=graph) as sess:
		saver.restore(sess, 'data/model_seeing.ckpt')
		Z = logits.eval(feed_dict={X: img})
		prediction=np.argmax(Z, axis=1)

		return prediction
###################################


##########################################
def predict_gestures(img):
	########## arquitetura da CNN: ResNet-34 - baseado no capítulo 13 do livro hands on machine learning
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
	n_outputs = 12

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

	# predict
	with tf.Session(graph=graph) as sess:
		saver.restore(sess, 'data/gestos/model_gestos.ckpt')
		Z = logits.eval(feed_dict={X: img})
		prediction=np.argmax(Z, axis=1)

	return prediction
###################################

##########################################
def predict_gestures2(img):
	########## arquitetura da CNN: ResNet-34 - baseado no capítulo 13 do livro hands on machine learning
	c1_depth = 6
	c1_ker_sz = 5
	c3_depth = 32
	c3_ker_sz = 6
	c5_depth = 120
	c5_ker_sz = 6

	num_hidden = 512
	n_labels = 12
	img_width = 128
	img_height = 96
	size = img_width, img_height
	n_channels = 1

	graph = tf.Graph()
	with graph.as_default():

		# Input data.
		tf_train_dataset = tf.placeholder(tf.float32, shape=(1, img_height, img_width, n_channels))
		tf_train_labels = tf.placeholder(tf.float32, shape=(None))
		# tf_test_dataset = tf.constant(test_dataset)

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
			# print(data.get_shape().as_list())

			conv = tf.nn.conv2d(data, c1_weights, [1, 1, 1, 1], padding='SAME')
			hidden = tf.nn.relu(conv + c1_biases)
			# print(conv.get_shape().as_list())

			pooled = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
			# print(pooled.get_shape().as_list())

			conv = tf.nn.conv2d(pooled, c3_weights, [1, 1, 1, 1], padding='SAME')
			hidden = tf.nn.relu(conv + c3_biases)
			pooled = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
			shape = pooled.get_shape().as_list()
			# print(shape)

			conv = tf.nn.conv2d(pooled, c5_weights, [1, 1, 1, 1], padding='SAME')
			hidden = tf.nn.relu(conv + c5_biases)
			pooled = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
			shape = pooled.get_shape().as_list()
			# print(shape)

			reshape = tf.reshape(pooled, [shape[0], shape[1] * shape[2] * shape[3]])
			hidden = tf.nn.relu(tf.matmul(reshape, fc_weights) + fc_biases)

			return tf.matmul(hidden, out_weights) + out_biases
	  
		# Training computation.
		logits = model(tf_train_dataset)
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
		
		optimizer = tf.train.AdagradOptimizer(0.001).minimize(loss)

		init = tf.global_variables_initializer()
		saver = tf.train.Saver()

	# predict
	with tf.Session(graph=graph) as sess:
		saver.restore(sess, 'data/checkpoints/model_gestos2.ckpt')
		Z = logits.eval(feed_dict={tf_train_dataset: img})
		# print(Z)
		# prediction=np.argmax(Z, axis=1)

	return Z
###################################

########## MAIN
if __name__ == '__main__':
	main()
########################