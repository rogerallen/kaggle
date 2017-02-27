#!/usr/bin/env python
"""
skeleton.py - a skeleton starting-point for python scripts by Roger Allen.

Any copyright is dedicated to the Public Domain.
http://creativecommons.org/publicdomain/zero/1.0/
You should add your own license here.

"""
import os
import sys
import logging
from optparse import OptionParser
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
from utils import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from sklearn.metrics import confusion_matrix
import pandas as pd

# ======================================================================

CURRENT_DIR   = os.getcwd()
DATA_HOME_DIR = CURRENT_DIR + '/data'
PATH          = DATA_HOME_DIR + '/'
RESULTS_PATH  = DATA_HOME_DIR + '/results/'

BATCH_SIZE = 64

# could simplify this bit
(val_classes, trn_classes, VAL_LABELS, TRN_LABELS,
 val_filenames, filenames, test_filenames) = get_classes(PATH)
VAL_BATCHES = get_batches(PATH+'valid', batch_size=BATCH_SIZE*2, shuffle=False)

CONV_FEAT     = load_array(PATH+'results/conv_feat.dat')
CONV_VAL_FEAT = load_array(PATH+'results/conv_val_feat.dat')

# ======================================================================

def validation_np_histogram(model):
    probs = model.predict(CONV_VAL_FEAT, batch_size=BATCH_SIZE*2)
    expected_labels = VAL_BATCHES.classes
    our_labels = np.argmax(probs, axis=1)
    histo = np.histogram(our_labels,range(11))[0]
    print "std %g\nhisto %s"%(np.std(histo),histo)

def validation_confusion(model):
    probs = model.predict(CONV_VAL_FEAT, batch_size=BATCH_SIZE*2)
    expected_labels = VAL_BATCHES.classes
    our_labels = np.argmax(probs, axis=1)
    cm = confusion_matrix(expected_labels, our_labels)
    plot_confusion_matrix(cm, VAL_BATCHES.class_indices)

# ======================================================================
def get_bn_layers(N,M,p,input_shape):
    return [
        MaxPooling2D(input_shape=input_shape),
        Flatten(),
        Dropout(p/2),
        #Dense(128, activation='relu'),
        Dense(N, activation='relu'),
        BatchNormalization(),
        Dropout(p/2),
        #Dense(128, activation='relu'),
        Dense(M, activation='relu'),
        BatchNormalization(),
        Dropout(p),
        Dense(10, activation='softmax')
    ]

def do_iter(model,lr,ep):
    model.optimizer.lr=lr
    model.fit(CONV_FEAT, TRN_LABELS, batch_size=BATCH_SIZE,
                 nb_epoch=ep,
                 validation_data=(CONV_VAL_FEAT, VAL_LABELS))
    validation_np_histogram(model)
    validation_confusion(model)

def run_model(N,M,p,lr0,ep0,save_label,load_label):
    print "run_model N:%d M:%d p:%f lr0:%g ep0:%d save_label:%s"%(N,M,p,lr0,ep0,save_label)
    model = Sequential(get_bn_layers(N,M,p,CONV_VAL_FEAT.shape[1:]))
    model.compile(Adam(lr=lr0),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
    if load_label != "":
        weights_filename='weights_%s.h5'%(load_label)
        print "loading %s"%(weights_filename)
        model.load_weights(RESULTS_PATH+weights_filename)
    do_iter(model, lr0, ep0)
    weights_filename='weights_%s.h5'%(save_label)
    print "saving %s"%(weights_filename)
    model.save_weights(RESULTS_PATH+weights_filename)

# ======================================================================
class Application(object):
    def __init__(self,argv):
        self.parse_args(argv)
        self.adjust_logging_level()

    def parse_args(self,argv):
        """parse commandline arguments, use config files to override
        default values. Initializes:
        self.options: a dictionary of your commandline options,
        self.args:    a list of the remaining commandline arguments
        """
        parser = OptionParser()
        # config file has verbosity level
        parser.add_option(
            "-v","--verbose",
            dest="verbose",
            action='count',
            default=0,
            help="Increase verbosity (can use multiple times)"
        )
        parser.add_option(
            "--dropout_rate",
            dest="dropout_rate",
            type="float",
            default=0.8,
            help="droput rate for model (default = 0.8)"
        )
        parser.add_option(
            "--N",
            dest="N",
            type="int",
            default=128,
            help="dense layer count N"
        )
        parser.add_option(
            "-M",
            dest="M",
            type="int",
            default=128,
            help="dense layer count M"
        )
        parser.add_option(
            "--lr0",
            dest="lr0",
            type="float",
            default=1e-5,
            help="initial learning rate"
        )
        parser.add_option(
            "--ep0",
            dest="ep0",
            type="int",
            default=10,
            help="initial epochs"
        )
        parser.add_option(
            "--label",
            dest="label",
            default="foo",
            help="label for filenames"
        )
        parser.add_option(
            "--load",
            dest="load_label",
            default="",
            help="weight label for loading"
        )
        self.options, self.args = parser.parse_args(argv)

    def adjust_logging_level(self):
        """adjust logging level based on verbosity option
        """
        log_level = logging.WARNING # default
        if self.options.verbose == 1:
            log_level = logging.INFO
        elif self.options.verbose >= 2:
            log_level = logging.DEBUG
        logging.basicConfig(level=log_level)

    def run(self):
        """The Application main run routine
        """
        logging.info("Options: %s, Args: %s" % (self.options, self.args))
        run_model(
            self.options.N,
            self.options.M,
            self.options.dropout_rate,
            self.options.lr0,
            self.options.ep0,
            self.options.label,
            self.options.load_label
        )
        return 0

# ======================================================================
def main(argv):
    """ The main routine creates and runs the Application.
    argv: list of commandline arguments without the program name
    returns application run status
    """
    app = Application(argv)
    return app.run()

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
