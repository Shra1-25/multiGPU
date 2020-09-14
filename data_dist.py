import numpy as np
import shutil
import os, glob
import h5py
import tensorflow as tf
import novograd as NovoGrad
datafile = glob.glob('../tfrecord_x1/*')
channels = [0,1,2,3,4,5,6,7]
#channels = [0,1,2]
granularity=1

# Mapping functions used to convert tfrecords to tf dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#@nvtx_tf.ops.trace(message='ExtractFromTFRecord', domain_name='DataLoading', grad_domain_name='BoostedJets')
def extract_fn(data):
    # extracts fields from TFRecordDataset
    feature_description = {
        #'X_jets': tf.io.FixedLenFeature([125*granularity*125*granularity*8], tf.float32),
        'm0': tf.io.FixedLenFeature([], tf.float32), 
        'pt': tf.io.FixedLenFeature([], tf.float32),
        'y': tf.io.FixedLenFeature([], tf.float32)
    }
    sample = tf.io.parse_single_example(serialized=data, features=feature_description)
    return sample
for i in range(1250):
    dataset = tf.data.TFRecordDataset(filenames=datafile[i], compression_type='GZIP', num_parallel_reads=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(extract_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    itr=dataset.as_numpy_iterator()
    source = datafile[i]
    if (itr.next()['y']==0):
        destination = "../tfrecord_x1_main/tfrecord_x1_0/BoostedJets_fullSample_x1_file-"+datafile[i].partition("-")[2]
        print(destination)
    else:
        destination = "../tfrecord_x1_main/tfrecord_x1_1/BoostedJets_fullSample_x1_file-"+datafile[i].partition("-")[2]
        print(destination)
    dest = shutil.copyfile(source, destination)
    print(str(i)+": "+str(itr.next())+" "+str(datafile[i]))
