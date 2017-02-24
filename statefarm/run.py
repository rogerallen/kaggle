import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
from utils import *
#from PIL import Image
#from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from sklearn.metrics import confusion_matrix
import pandas as pd

current_dir = os.getcwd()
LESSON_HOME_DIR = current_dir
DATA_HOME_DIR = current_dir+'/data'
categories = sorted([os.path.basename(x) for x in glob(DATA_HOME_DIR+'/train/*')])

#Set path to sample/ path if desired
path = DATA_HOME_DIR + '/'
#path = DATA_HOME_DIR + '/sample/'

test_path = DATA_HOME_DIR + '/test/' #We use all the test data
results_path=DATA_HOME_DIR + '/results/'
train_path=path + '/train/'
valid_path=path + '/valid/'

histories = {}

vgg = Vgg16()

batch_size=64

model=vgg.model
last_conv_idx = [i for i,l in enumerate(model.layers) if type(l) is Convolution2D][-1]
conv_layers = model.layers[:last_conv_idx+1]
conv_model = Sequential(conv_layers)
(val_classes, trn_classes, val_labels, trn_labels,
    val_filenames, filenames, test_filenames) = get_classes(path)

conv_feat = load_array(path+'results/conv_feat.dat')
conv_val_feat = load_array(path+'results/conv_val_feat.dat')

def get_bn_layers(p):
    return [
        MaxPooling2D(input_shape=conv_layers[-1].output_shape[1:]),
        Flatten(),
        Dropout(p/2),
        #Dense(128, activation='relu'),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(p/2),
        #Dense(128, activation='relu'),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(p),
        Dense(10, activation='softmax')
    ]

p=0.8 #-- wow isn't this high?

bn_model = Sequential(get_bn_layers(p))
bn_model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# starting with super-small lr first
bn_model.optimizer.lr=0.00001
bn_model.fit(conv_feat, trn_labels, batch_size=batch_size, nb_epoch=3,
             validation_data=(conv_val_feat, val_labels))

bn_model.optimizer.lr=0.1
bn_model.fit(conv_feat, trn_labels, batch_size=batch_size, nb_epoch=10,
             validation_data=(conv_val_feat, val_labels))

bn_model.optimizer.lr=0.01
bn_model.fit(conv_feat, trn_labels, batch_size=batch_size, nb_epoch=10,
             validation_data=(conv_val_feat, val_labels))

bn_model.optimizer.lr=0.001
bn_model.fit(conv_feat, trn_labels, batch_size=batch_size, nb_epoch=10,
             validation_data=(conv_val_feat, val_labels))

latest_weights_filename='test1.h5'
bn_model.save_weights(results_path+latest_weights_filename)
