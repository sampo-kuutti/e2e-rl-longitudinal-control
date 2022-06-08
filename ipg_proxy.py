# Author: Sampo Kuutti
# Proxy model for estimating response from IPG CarMaker simulations
# ...

import proxy_model
import tensorflow as tf
import numpy as np
import argparse
import os

fpath = 'S:/Research/safeav/Sampo/condor-a2c/test/'  # use same directory path

NUM_INPUTS = 5
MODEL_FILE = 'model-step-981000-val-7.87466e-05.ckpt'
DATA_DIR = './data/'
LOG_DIR = fpath



class IpgProxy(object):
    """implements the ipg proxy model for emulating the IPG CarMaker simulation environment"""

    def __init__(self, model_file=MODEL_FILE, num_inputs=NUM_INPUTS, data_dir=DATA_DIR, log_dir=LOG_DIR):
        # set up tf session and model
        #args = get_arguments()
        self.num_inputs = num_inputs
        ipg_graph = tf.Graph()
        with ipg_graph.as_default():   # create a new graph and sess for ipg_proxy
            self.model = proxy_model.SupervisedModel()
        self.sess_1 = tf.Session(graph=ipg_graph)
        with self.sess_1.as_default():
            with ipg_graph.as_default():
                saver = tf.train.Saver()
                checkpoint_path = os.path.join(log_dir, model_file)
                saver.restore(self.sess_1, checkpoint_path)
        print('ipg_proxy: Restored model: %s' % model_file)

    def inference(self, x):
        with self.sess_1.as_default():
            x = np.reshape(x, (1, self.num_inputs))       # reshape to a valid shape for input to nn
            y = self.model.y.eval(feed_dict={self.model.x: x})      # output network prediction
        return y


