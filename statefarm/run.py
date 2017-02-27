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

def validation_np_histogram():
    probs = bn_model.predict(conv_val_feat, batch_size=batch_size)
    expected_labels = val_batches.classes
    our_labels = np.argmax(probs, axis=1)
    print np.histogram(our_labels,range(11))[0]
    
def validation_confusion():
    probs = bn_model.predict(conv_val_feat, batch_size=batch_size)
    expected_labels = val_batches.classes
    our_labels = np.argmax(probs, axis=1)
    cm = confusion_matrix(expected_labels, our_labels)
    plot_confusion_matrix(cm, val_batches.class_indices)

batch_size=64

(val_classes, trn_classes, val_labels, trn_labels,
    val_filenames, filenames, test_filenames) = get_classes(path)
val_batches = get_batches(path+'valid', batch_size=batch_size, shuffle=False)

conv_feat = load_array(path+'results/conv_feat.dat')
conv_val_feat = load_array(path+'results/conv_val_feat.dat')

def get_bn_layers(p,input_shape):
    return [
        MaxPooling2D(input_shape=input_shape),
        Flatten(),
        Dropout(p/2),
        #Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(p/2),
        #Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(p),
        Dense(10, activation='softmax')
    ]

p=0.8 #-- wow isn't this high?

bn_model = Sequential(get_bn_layers(p,conv_val_feat.shape[1:]))
bn_model.compile(Adam(lr=0.000001), loss='categorical_crossentropy', metrics=['accuracy'])

# starting with super-small lr first
def do_iter(lr,ep):
    bn_model.optimizer.lr=lr
    bn_model.fit(conv_feat, trn_labels, batch_size=batch_size, nb_epoch=ep,
                 validation_data=(conv_val_feat, val_labels))
    validation_np_histogram()
    validation_confusion()

do_iter(0.0000001, 10)
do_iter(0.001, 10)
do_iter(0.0001, 10)

latest_weights_filename='test1.h5'
bn_model.save_weights(results_path+latest_weights_filename)
