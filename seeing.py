import argparse
from threading import Thread
from queue import Queue
import sys, os
import cv2
import numpy as np
import face_detection_utilities as fdu
import time

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# algumas variáveis necessárias
windowsName = 'Preview Screen'
REC_COLOR = (0, 255, 0)
FACE_SHAPE = (48, 48)

# inputs
parser = argparse.ArgumentParser(description='Módulo de visão do EuRobo')
parser.add_argument('-test', help=('Dado o caminho de uma imagem de teste, o app tenta prever a expressão facial.'))
parser.add_argument('-train', help=('Captura 10 imagens da webcam para serem usadas em treinamento de CNN. Opções: 0-sorridente, 1-triste, 2-surpreso, 3-medo, 4-raiva, 5-normal .'))
args = parser.parse_args()

emo = ['Sorridente', 'Triste', 'Surpreso', 'Medo', 'Raiva', 'Neutro', 'Maluco']


########## MAIN FUNCTION
def main():
    print("Inicializando módulo de visão bot")

    # inicio captura
    capture = cv2.VideoCapture(0)
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
    
    showScreenAndDectect(capture)



def init_seeing():
    # inicio captura
    capture = cv2.VideoCapture(0)
    if not capture:
        print("Bro, não tá rolando essa câmera, algo errado\n")
        sys.exit(1)
    else:
        print("Estou conseguindo te ver\n")
        showScreenAndDectect(capture)


########## Detectando faces, recortando, tratando e mandando pro algoritmo de ML
def showScreenAndDectect(capture):
    while (True):

        # começa a capturar 
        flag, frame = capture.read()

        # coordenadas do rosto no esquema [0,0,0,0]
        faceCoordinates = fdu.getFaceCoordinates(frame)
        
        if faceCoordinates is not None:
            
            # desenha um retangulo ao redor do rosto
            cv2.rectangle(np.asarray(frame), (faceCoordinates[0], faceCoordinates[1]), (faceCoordinates[2], faceCoordinates[3]), REC_COLOR, thickness=2)

            # crap apenas na regiao da face 
            face = frame[faceCoordinates[1]:faceCoordinates[3], faceCoordinates[0]:faceCoordinates[2]]

            # resize pro tamanho esperado pelo algoritmo
            face_scaled = cv2.resize(face, (48,48))

            # converte pra grayscale
            face_img = cv2.cvtColor(face_scaled, cv2.COLOR_BGR2GRAY)

            # mostra os frames numa janela
            cv2.startWindowThread()
            cv2.imshow(windowsName, frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # descobre a expressão facial com base na imagem (converto pro modelo que o tf está esperando)
            img = np.reshape(face_img, (-1, 2304))

            test_emo = predict(img)
            index_emo = test_emo[0]
            print (emo[index_emo])

            # 3 vezes por segundo
            time.sleep(0.33)

        return True

    capture.release()
    cv2.destroyAllWindows()


########## Treinando o bichinho 
def getFacesToTrain(capture, emo, iteractions):
    count = 0
    while count < iteractions:

        # inicializo captura
        flag, frame = capture.read()

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
            cv2.imwrite('Data/images/{0}/img_{1}.jpg'.format(emo, count), face_img)

            count = count + 1
            time.sleep(0.33)
        print(count)


    print("Pronto. Capturadas as imagens para a expressão:", emo)
    sys.exit(0)
##########################################


def predict(img):
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
        saver.restore(sess, 'Data/model_seeing.ckpt')
        Z = logits.eval(feed_dict={X: img})
        prediction=np.argmax(Z, axis=1)

        return prediction
###################################


########## MAIN
if __name__ == '__main__':
    main()
########################