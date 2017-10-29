import numpy as np
import cv2

import glob
import random
import numpy as np
import os

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Emotion list
LBPHF = cv2.face.createLBPHFaceRecognizer() #Initialize fisher face classifier
download_dir = 'C:\\Users\\Cezary Kujawski\\Downloads\\'
labels_regexp = '{}extended-cohn-kanade-images\\Emotion\\*\\*\\*.txt'.format(download_dir)
images_dir = '{}extended-cohn-kanade-images\\cohn-kanade-images'.format(download_dir)

data = {}

def get_files(): #Define function to get file list, randomly shuffle it and split 80/20
    labels = glob.glob(labels_regexp)
    images = map(lambda file:file.split('\\')[-1].strip('_emotion.txt').split('_'), labels)
    files_with_labels = {images_dir + '\\' + image[0] + '\\' + image[1] + '\\' + image[0] + '_' + image[1] + '_' + image[2] + '.png'
				:int(read_file(label).strip().strip('\n')[0])
					for image, label in zip(images, labels)}
    files_with_labels_list = [(file, files_with_labels[file]) for file in files_with_labels]
    random.shuffle(files_with_labels_list)
    training = dict(files_with_labels_list[:int(len(files_with_labels_list)*0.8)]) #get first 80% of file list
    prediction = dict(files_with_labels_list[-int(len(files_with_labels)*0.2):]) #get last 20% of file list
    return training, prediction

def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    training, prediction = get_files()
    #Append data to training and prediction list, and generate labels 0-7
    for item in training:
        image = cv2.imread(item) #open image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
        training_data.append(gray) #append image array to training data list
        training_labels.append(training[item])

    for item in prediction: #repeat above process for prediction set
        image = cv2.imread(item)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        prediction_data.append(gray)
        prediction_labels.append(prediction[item])

    return training_data, training_labels, prediction_data, prediction_labels

def run_recognizer():
    training_data, training_labels, prediction_data, prediction_labels = make_sets()
    
    print ("training fisher face classifier")
    print ("size of training set is:", len(training_labels), "images")
    LBPHF.train(training_data, np.asarray(training_labels)) #train it

    print ("predicting classification set")
    cnt = 0
    correct = 0
    incorrect = 0
    for image in prediction_data:
        pred, conf = LBPHF.predict(image) #predict emotion
        if pred == prediction_labels[cnt]: #validate it
            correct += 1
            cnt += 1
        else:
            incorrect += 1
            cnt += 1
    return ((100*correct)/(correct + incorrect))
    
def read_file(file):
    with open(file, 'r') as f:
        return f.read()

#labels_path = 'C:\Users\Cezary Kujawski\Downloads\extended-cohn-kanade-images\Emotion\\'



#Now run it
metascore = []
for i in range(0,15):
    correct = run_recognizer()
    print ("got", correct, "percent correct!")
    metascore.append(correct)

print ("\n\nend score:", np.mean(metascore), "percent correct!")
'''
#Video part
cap = cv2.VideoCapture('./../widea/laughter_480.mkv') #open video

while(cap.isOpened()):
    ret, frame = cap.read() #read frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #make it gray

    font = cv2.FONT_HERSHEY_SIMPLEX
    pred, conf = LBPHF.predict(gray) #predict emotion
    
    cv2.putText(gray,"emotion: " + emotions[pred],(200,50), font, 1, (200,255,155), 2, cv2.LINE_AA)
    
    cv2.imshow('frame',gray)
    
    if cv2.waitKey(25) & 0xFF == ord('q'): #load another frame after 25ms
        break

cap.release()
cv2.destroyAllWindows()

'''