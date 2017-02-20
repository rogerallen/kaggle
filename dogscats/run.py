#!/usr/bin/env python
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
from utils import *
from vgg16bn import Vgg16BN
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

current_dir = os.getcwd()
LESSON_HOME_DIR = current_dir
DATA_HOME_DIR = current_dir+'/data'

# or '/sample/'
path = DATA_HOME_DIR + '/'
test_path = DATA_HOME_DIR + '/test/' #We use all the test data
results_path=DATA_HOME_DIR + '/results/'
train_path=path + '/train/'
valid_path=path + '/valid/'

vgg = Vgg16BN()

batch_size = 64

# ADJUST THIS
no_of_epochs = 30
latest_weights_filename = None
run_index = 23
#learning_rate = 0.0001 # was 0.01.  reduced at 23 again at 28

# augment images
gen = image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True)

batches = vgg.get_batches(train_path, gen, batch_size=batch_size)
val_batches = vgg.get_batches(valid_path, batch_size=batch_size*2)
vgg.finetune(batches)

#vgg.model.optimizer.lr = learning_rate
INIT_LR=0.2
EPOCHS_DROP=5.0
DROP=0.5

def step_decay(epoch, initial_lrate = INIT_LR, epochs_drop = EPOCHS_DROP, drop = DROP):
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

if latest_weights_filename != None:
    print "loading %s"%(results_path+latest_weights_filename)
    vgg.model.load_weights(results_path+latest_weights_filename)

filepath=results_path+"run-%02d-weights-{epoch:02d}-{val_acc:.2f}.hdf5"%(run_index)
history_filepath=results_path+"run-%02d-history.csv"%(run_index)

checkpoint = ModelCheckpoint(filepath,
                             # seemed ot get worse results with val_acc
                             #monitor='val_acc', mode='max',
                             monitor='val_loss', mode='min',
                             verbose=1,
                             save_weights_only=True, save_best_only=True)
lr_scheduler = LearningRateScheduler(step_decay)
callbacks = [checkpoint,lr_scheduler]

history = vgg.fit(batches, val_batches, no_of_epochs, callbacks)

val_batches, probs = vgg.test(valid_path, batch_size = batch_size)
filenames = val_batches.filenames
expected_labels = val_batches.classes #0 or 1

#Round our predictions to 0/1 to generate labels
our_predictions = probs[:,0]
our_labels = np.round(1-our_predictions)

cm = confusion_matrix(expected_labels, our_labels)
print "Confusion Matrix"
print cm
