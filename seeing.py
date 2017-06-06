import argparse
from threading import Thread
from queue import Queue
import sys, os
import cv2
import numpy as np
import face_detection_utilities as fdu
import time

# some variables
windowsName = 'Preview Screen'
REC_COLOR = (0, 255, 0)
FACE_SHAPE = (48, 48)

# input commands
parser = argparse.ArgumentParser(description='A live emotion recognition from webcam')
parser.add_argument('-test', help=('Given the path of testing image, the program will predict the result of the image.'))
parser.add_argument('-trainHappy', help=('Use images from webcam to train the model.'))
args = parser.parse_args()

#emotions = 1, 2, 3, 4, 5, 6
emo = ['Sorridente', 'Triste', 'Surpreso', 'Medo', 'Raiva', 'Neutro']


########## MAIN FUNCTION
def main():
    print("Inicializando o bot")
    
    print("Pressione o número correspondente a emoção que você quer treinar:")
    print("1: feliz, 2: triste, 3: normal, 4: com raiva, 5: surpreso, 6: com medo")
    inputdata = input()
    if inputdata == 1:
        print("OK, as you wish my friend...")

    # handle args
    if args.test is not None:
        img = cv2.imread(args.test)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, FACE_SHAPE)
        print(class_label[result[0]])
        sys.exit(0)

    # start capturing
    capture = cv2.VideoCapture(0)
    if not capture:
        print("Failed to capture video streaming ")
        sys.exit(1)
    else:
        print("Successed to capture video streaming")
    
    showScreenAndDectect(capture, training_label = inputdata)

########## Detecting faces
def showScreenAndDectect(capture, training_label):
    while (True):

        #start capturing 
        flag, frame = capture.read()

        #face coordinates [0,0,0,0]
        faceCoordinates = fdu.getFaceCoordinates(frame)
        
        if faceCoordinates is not None:
            
            #draw rectangle around the face
            cv2.rectangle(np.asarray(frame), (faceCoordinates[0], faceCoordinates[1]), (faceCoordinates[2], faceCoordinates[3]), REC_COLOR, thickness=2)

            #crop the face using the same coordinates 
            face = frame[faceCoordinates[1]:faceCoordinates[3], faceCoordinates[0]:faceCoordinates[2]]

            #scale it down to fit the model parameters
            face_scaled = cv2.resize(face, (48,48))

            #convert to grayscale to work with just one channel
            face_img = cv2.cvtColor(face_scaled, cv2.COLOR_BGR2GRAY)

            # save image to file in case the user choose an emotion to train
            emo_id = int(training_label)
            if emo_id > 0:
            	print("Salvando imagem no label:", emo_id)
            	cv2.imwrite('Data/images/%04i/img.jpg' %emo_id, face_img)


            #show the frames in a window
            cv2.startWindowThread()
            cv2.namedWindow(windowsName, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(windowsName, cv2.WND_PROP_FULLSCREEN, cv2.WND_PROP_FULLSCREEN)
            cv2.imshow(windowsName, frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # predict the emotion using each frame
            # input_img = np.expand_dims(face_img, axis=0)
            # input_img = np.expand_dims(input_img, axis=0)

            # result = model.predict(input_img)[0]
            # index = np.argmax(result)

            # if max(result) > 0.5:
            #     #print(face_img.shape)
            #     emo[index]
            #     print (emo[index], 'prob:', max(result))

            # run the loop 3 times per second
            time.sleep(0.33)

    capture.release()
    cv2.destroyAllWindows()


########## MAIN
if __name__ == '__main__':
    main()