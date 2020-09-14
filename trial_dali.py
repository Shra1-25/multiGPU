import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.plugin.tf as dali_tf
import numpy as np
import os, glob
import h5py
import tensorflow as tf
from novograd import NovoGrad
import horovod.tensorflow.keras as hvd
from sklearn.metrics import roc_curve, auc
import tensorflow.keras as keras
import math
import time
import datetime
from tensorflow.keras.mixed_precision import experimental as mixed_precision

flag_device='gpu' # change this flag to gpu if you want to use gpu instead
BATCH_SZ = 32*50
train_sz = 32*80356
valid_sz = 32*12362
test_sz  = 32*24725

hvd.init()
valid_steps = valid_sz // (BATCH_SZ*hvd.size())
test_steps  = test_sz  // (BATCH_SZ*hvd.size())

channels = [0,1,2,3,4,5,6,7]
#channels = [0,1,2]
granularity=1
classes = 2

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        print("\n Timestamp: "+str(tf.cast(tf.timestamp(),tf.float64)))

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training parameters.')
    parser.add_argument('-e', '--epochs', default=20, type=int, help='Number of training epochs.')
    parser.add_argument('-l', '--lr_init', default=5.e-4, type=float, help='Initial learning rate.')
    parser.add_argument('-b', '--resblocks', default=3, type=int, help='Number of residual blocks.')
    parser.add_argument('-c', '--cuda', default=0, type=int, help='Which gpuid to use.')
    parser.add_argument('-a', '--load_epoch', default=0, type=int, help='Which epoch to start training from')
    parser.add_argument('-s', '--save_dir', default='MODELS', help='Directory with saved weights files')
    parser.add_argument('-n', '--name', default='', help='Name of experiment')
    parser.add_argument('--warmup-epochs', type=float, default=5, help='number of warmup epochs')
    args = parser.parse_args()

    lr_init = args.lr_init
    resblocks = args.resblocks
    epochs = args.epochs
    expt_name = 'BoostedJets-opendata_ResNet_blocks%d_x1_epochs%d'%(resblocks, epochs)
    expt_name = expt_name + '-' +  datetime.date.strftime(datetime.datetime.now(),"%Y%m%d-%H%M%S")
    if len(args.name) > 0:
        expt_name = args.name
    if not os.path.exists('MODELS/' + expt_name):
        os.mkdir('MODELS/' + expt_name)
# only set `verbose` to `1` if this is the root worker. Otherwise, it should be zero.
if hvd.rank() == 0:
    verbose = 1
else:
    verbose = 0

def LR_Decay(epoch):
    drop = 0.5
    epochs_drop = 10
    lr = lr_init * math.pow(drop, math.floor((epoch+1)/epochs_drop))
    return lr

def restart_epoch(args):
    epoch = 0
    for try_epoch in range(args.epochs, 0, -1):
        if os.path.exists(expt_name.format(epoch=try_epoch)):
            epoch = try_epoch
            break

    return epoch

class FalconPipeline(Pipeline):
    def __init__(self, img_file_list,  num_shards, shard_id, batch_size, num_threads, device_id, flag_device):
        super(FalconPipeline, self).__init__(batch_size, num_threads, device_id, seed = 12)
        self.device = flag_device
        self.input_img = ops.FileReader(file_root = "../tfrecord_x1_main", num_shards=num_shards, shard_id=shard_id, file_list=img_file_list, random_shuffle = True, initial_fill = 21)
        self.decode = ops.ImageDecoder(device = 'mixed' if self.device == 'gpu' else 'cpu', output_type = types.RGB)
        # to change according to your needs
        self.rrc = ops.RandomResizedCrop(
                device=self.device,
                size=(125,125),
                random_area=[0.8, 0.8]
            )
    def define_graph(self):
        imgs, labels = self.input_img()
        images = self.decode(imgs)
        output = self.rrc(images)
        return (output, labels)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
USE_XLA = True
if USE_XLA:
    tf.config.optimizer.set_jit(USE_XLA)

def create_resnet():
    # Build network
    import keras_resnet_single as networks
    resnet = networks.ResNet.build(len(channels), 3, [16,32], (125*granularity,125*granularity,len(channels)), granularity)
    # Load saved weights, if indicated
    #if args.load_epoch != 0:
    #    directory = args.save_dir
    #    if args.save_dir == '':
    #        directory = expt_name
    if args.load_epoch != 0:
        directory = args.save_dir
        if args.save_dir == '':
            directory = expt_name
        model_name = glob.glob('../MODELS/%s/epoch%02d-*.hdf5'%(directory,0 ))[0]
        #assert len(model_name) == 2
        #model_name = model_name[0].split('.hdf5')[0]+'.hdf5'
        print('Loading weights from file:', model_name)
        resnet.load_weights(model_name)
    #opt = keras.optimizers.Adam(lr=lr_init, epsilon=1.e-5) # changed eps to match pytorch value
    #opt = keras.optimizers.SGD(lr=lr_init * hvd.size())
    opt = NovoGrad(learning_rate=lr_init * hvd.size())
    #Wrap the optimizer in a Horovod distributed optimizer -> uses hvd.DistributedOptimizer() to compute gradients.
    opt = hvd.DistributedOptimizer(opt)

    #For Horovod: We specify `experimental_run_tf_function=False` to ensure TensorFlow
    #resnet.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'], experimental_run_tf_function = False)
    #resnet.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    resnet.summary()
    return resnet

if __name__ == '__main__':
    decay = ''
    #print(">> Input file:",datafile)
    expt_name = '%s_%s'%(decay, expt_name)
    for d in ['MODELS', 'METRICS']:
        if not os.path.isdir('%s/%s'%(d, expt_name)):
            os.makedirs('%s/%s'%(d, expt_name))


pipe = FalconPipeline("../tfrecord_x1_main/file_list.txt", 2, 0, BATCH_SZ, 8, 0, flag_device) # device_id is
pipe.build()
resnet = create_resnet()

callbacks_list = []
callbacks_list.append(myCallback())
callbacks_list.append(hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=args.warmup_epochs, verbose=verbose))
callbacks_list.append(hvd.callbacks.LearningRateScheduleCallback(start_epoch=args.warmup_epochs, multiplier=LR_Decay))
callbacks_list.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
callbacks_list.append(hvd.callbacks.MetricAverageCallback())
resume_from_epoch = 0
#checkpointing should only be done on the root worker.
if hvd.rank() == 0:
    callbacks_list.append(keras.callbacks.ModelCheckpoint('MODELS/' + expt_name + '/epoch{epoch:02d}-{val_loss:.2f}.hdf5', verbose=verbose, save_best_only=False))#, save_weights_only=True)
    callbacks_list.append(keras.callbacks.TensorBoard(args.save_dir))
resume_from_epoch = restart_epoch(args)
resume_from_epoch = hvd.broadcast(resume_from_epoch, 0)    

daliop = dali_tf.DALIIterator()
shapes = [(BATCH_SZ, 125, 125, 8), (BATCH_SZ, 2)]
dtypes = [tf.float32, tf.int32]
# Create TF dataset
out_dataset = dali_tf.DALIDataset(
             pipeline=pipe,
             batch_size=BATCH_SZ,
             shapes=shapes,
             dtypes=dtypes,
             device_id=0)
opt = NovoGrad(learning_rate=lr_init * hvd.size())
opt = hvd.DistributedOptimizer(opt)
resnet.compile(
             optimizer=opt,
             loss='binary_crossentropy',
             metrics=['accuracy'],experimental_run_tf_function = False)
# Train using DALI dataset
history = resnet.fit(
            out_dataset,
            steps_per_epoch= train_sz // (BATCH_SZ*hvd.size()),
            epochs=epochs,
            callbacks=callbacks_list,
            verbose=verbose,
            workers=hvd.size(),
            initial_epoch=resume_from_epoch,
           # validation_data=val_data,
           # validation_steps = 3 * valid_steps,
            use_multiprocessing=True)
