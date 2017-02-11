#!/usr/bin/env python
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
from utils import *
from vgg16 import Vgg16
from sklearn.metrics import confusion_matrix

current_dir = os.getcwd()
LESSON_HOME_DIR = current_dir
DATA_HOME_DIR = current_dir+'/data'

# or '/sample/'
path = DATA_HOME_DIR + '/' 
test_path = DATA_HOME_DIR + '/test/' #We use all the test data
results_path=DATA_HOME_DIR + '/results/'
train_path=path + '/train/'
valid_path=path + '/valid/'

vgg = Vgg16()

batch_size = 64

# ADJUST THIS ?cmdline options?
no_of_epochs = 10
load_index = 0

latest_weights_filename='ft%d.h5'%(load_index)
epoch_offset=load_index+1

batches = vgg.get_batches(train_path, batch_size=batch_size)
val_batches = vgg.get_batches(valid_path, batch_size=batch_size*2)
vgg.finetune(batches)

vgg.model.optimizer.lr = 0.01

print "loading %s"%(results_path+latest_weights_filename)
vgg.model.load_weights(results_path+latest_weights_filename)

for epoch in range(no_of_epochs):
    print "Running epoch: %d" % (epoch + epoch_offset)
    vgg.fit(batches, val_batches, nb_epoch=1)
    latest_weights_filename = 'ft%d.h5' % (epoch + epoch_offset)
    vgg.model.save_weights(results_path+latest_weights_filename)

print "Completed %s fit operations" % no_of_epochs

val_batches, probs = vgg.test(valid_path, batch_size = batch_size)
filenames = val_batches.filenames
expected_labels = val_batches.classes #0 or 1

#Round our predictions to 0/1 to generate labels
our_predictions = probs[:,0]
our_labels = np.round(1-our_predictions)

cm = confusion_matrix(expected_labels, our_labels)
print "Confusion Matrix"
print cm
