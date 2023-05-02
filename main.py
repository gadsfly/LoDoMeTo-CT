import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    concatenate,
    Conv2DTranspose,
    Dense
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import argparse
import numpy as np
import pickle
import time
import os
from pathlib import Path
import h5py
from skimage.transform import rescale, resize, iradon

from racts_utils import load_sim_data 

max_sinogram_intensity = 500
max_scan_intensity = 5.16


#### SAVING AND LOADING

def load_run_h5(hdf5_file_path):
    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
            # Load the image
        input_sinos = hdf5_file['input_sinos'][:]
        input_scans = hdf5_file['input_scans'][:]
        sampled_sinos = hdf5_file['sampled_sinos'][:]
        out_sinos = hdf5_file['out_sinos'][:]
        out_scans = hdf5_file['out_scans'][:]
        
        return {
            'input_sinos' : input_sinos,
            'input_scans' : input_scans,
            'sampled_sinos' : sampled_sinos,
            'out_sinos' : out_sinos,
            'out_scans' : out_scans
            
        }

def save_run_h5(unet, hdf5_file_path, n = 5):
    sm_input_sino = Model(inputs=unet.inputs,
                              outputs=unet.get_layer(name='input_sino').output)

    sm_input_scan = Model(inputs=unet.inputs,
                              outputs=unet.get_layer(name='input_scan').output)

    sm_sampled = Model(inputs=unet.inputs,
                              outputs=unet.get_layer(name='sample').output)

    val_ds = get_ds('val', version=2, n =n, ds_bsz=n)

    input_ =  next(iter(val_ds))[0]
    input_sinos = sm_input_sino(input_).numpy()
    input_scans = sm_input_scan(input_).numpy() 
    sampled_sinos = sm_sampled(input_).numpy() 
    out_sinos, out_scans = unet(input_)
    out_sinos = out_sinos.numpy()
    out_scans = out_scans.numpy()

    
    with h5py.File(hdf5_file_path, 'w') as hdf5_file:
        # Save the image
        hdf5_file.create_dataset('input_sinos', data=input_sinos)
        hdf5_file.create_dataset('input_scans', data=input_scans)
        hdf5_file.create_dataset('sampled_sinos', data=sampled_sinos)
        hdf5_file.create_dataset('out_sinos', data=out_sinos)
        hdf5_file.create_dataset('out_scans', data=out_scans)

def iradon_tf(sinogram):
    
    def _pre_iradon_wrapper(sinogram_):
        def _iradon_wrapper(s):
            theta = np.linspace(0., 180., 512, endpoint=False)
            s = s.reshape((512,512))
            return iradon(s, theta=theta, circle=False)
        
        sinogram_ = sinogram_.numpy()
        bsz = sinogram_.shape[0]
        h = sinogram_.shape[1]
        w = sinogram_.shape[2]
        sinogram_ = sinogram_.reshape(bsz, h*w)
        recon_ = np.apply_along_axis(_iradon_wrapper, axis=(1), arr=sinogram_)
        recon_ = recon_.reshape(bsz, 362, 362, 1)
        return recon_

    recon_scan = tf.py_function(_pre_iradon_wrapper, [sinogram], tf.float32)
    recon_scan.set_shape([None, 362, 362, 1])
    
    return recon_scan

class InverseRadonTransform(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        recon_scan = iradon_tf(x)
        return recon_scan


class ChangeSamplingRate(tf.keras.layers.Layer):
    def __init__(self, sr=0.1, name=None):
        super(ChangeSamplingRate, self).__init__(name=name)
        self.sr = sr
    
        indices = np.arange(0, 512, 1/self.sr).astype(int)
        print("effective sr: {:.2f}".format(len(indices)/512))  
        
    def call(self, x):
        sampled_sinogram = self.sample_sino(x)
        return sampled_sinogram

    def sample_sino(self, sinogram):

        # get number of columns in sinogram (equivalent to number of samples taken in scan)
        n_s = sinogram.shape[2] # 2 is the right dimension for BATCHED INPUT; otherwise 1

        # for given sampling rate, get columns that will not be sampled
        indices = np.arange(0, n_s, 1/self.sr).astype(int)
        theta_inds = np.arange(0, n_s)
        zero_inds = list(set(theta_inds) - set(indices)) # indices to be set to 0
        
        # set non-sampled columns to 0
        mask = [tf.one_hot(col_num*tf.ones((sinogram.shape[1], ), dtype=tf.int32), sinogram.shape[2])
                for col_num in zero_inds]
        mask = tf.reduce_sum(mask, axis=0)
        mask = tf.cast(tf.logical_not(tf.cast(mask, tf.bool)), tf.float32)
        mask = tf.expand_dims(mask, 0)
        mask = tf.expand_dims(mask, -1)
        result = sinogram * mask

        return result
    
def mse(tens1, tens2):
    return tf.reduce_mean(tf.square(tens1 - tens2))

def psnr(tens1, tens2):
    return tf.image.psnr(tens1, tens2, max_val=1.0)

def ssim(tens1, tens2):
    return tf.image.ssim(tens1, tens2, max_val=1.0, filter_size=4,
                          filter_sigma=1.5, k1=0.01, k2=0.03)

class MetricLayer(tf.keras.layers.Layer):
    def __init__(self, mse, ssim, psnr, scan_metrics=True):
        super().__init__()
        self.mse = mse
        self.ssim = ssim
        self.psnr = psnr
        self.scan_metrics = scan_metrics

    def call(self, inputs):
        
        if self.scan_metrics:
            input_sino, input_scan, out_sino, out_scan = inputs

            # Compute metrics
            self.add_metric(self.ssim(input_sino, out_sino), name='ssim_sino')
            self.add_metric(self.psnr(input_sino, out_sino), name='psnr_sino')

            self.add_metric(self.mse(input_scan, out_scan), name='mse_scan')
            self.add_metric(self.ssim(input_scan, out_scan), name='ssim_scan')
            self.add_metric(self.psnr(input_scan, out_scan), name='psnr_scan')
            
            return [out_sino, out_scan]
        else:
            
            input_sino, out_sino = inputs

            # Compute metrics
            self.add_metric(self.ssim(input_sino, out_sino), name='ssim_sino')
            self.add_metric(self.psnr(input_sino, out_sino), name='psnr_sino')        

        return out_sino
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "mse" : self.mse,
            "ssim" : self.ssim,
            "psnr" : self.psnr,
            "scan_metrics" : self.scan_metrics
        })
        return config
    
class LossLayer(tf.keras.layers.Layer):
    def __init__(self, loss_func):
        super().__init__()
        self.loss_func = loss_func

    def call(self, inputs):
        input_sino, out_sino = inputs
#         input_sino, out_sino = inputs
#         print('minmax', tf.math.reduce_min(input_sino), tf.math.reduce_max(out_sino))
        # Compute loss
        loss = self.loss_func(input_sino, out_sino)

        # Add loss
        self.add_loss(loss)

        # Do not modify the outputs
        return out_sino
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "loss_func" : self.loss_func,
        })
        return config
   


def unet_model(input_shape_sino = (512, 512, 1), 
               input_shape_scan = (362, 362, 1),
               irt_mode = 'classic',
               base_filter=64, kernel_size=3,strides=(1, 1), sr=0.1):
    """
    sr : sampling_rate
    """
    
    input_sino_ = Input(shape=input_shape_sino)
    input_scan_ = Input(shape=input_shape_scan)
    
    # normalize
    input_sino = tf.keras.layers.Rescaling(1./max_sinogram_intensity, offset=0.0, name = 'input_sino')(input_sino_)
    input_scan = tf.keras.layers.Rescaling(1./max_scan_intensity, offset=0.0, name = 'input_scan')(input_scan_)
#     print(f"input_sino shape: {input_sino.shape}, input_scan shape: {input_scan.shape}")

    #physical layer
    x_sino = ChangeSamplingRate(sr, name='sample')(input_sino) # sr --> percentage sampled
    
    # down
    con1 = Conv2D(base_filter, kernel_size, strides, padding='same', activation="relu")(x_sino)
    con1 = Conv2D(base_filter, kernel_size, strides, padding='same', activation="relu")(con1)
    x_sino = MaxPooling2D(pool_size=(2, 2))(con1)
    con2 = Conv2D(base_filter*2, kernel_size, strides, padding='same', activation="relu")(x_sino)
    con2 = Conv2D(base_filter*2, kernel_size, strides, padding='same', activation="relu")(con2)
    x_sino = MaxPooling2D(pool_size=(2, 2))(con2)
    con3 =  Conv2D(base_filter*4, kernel_size, strides, padding='same', activation="relu")(x_sino)
    con3 =  Conv2D(base_filter*4, kernel_size, strides, padding='same', activation="relu")(con3)
    x_sino = MaxPooling2D(pool_size=(2, 2))(con3)
    con4 =  Conv2D(base_filter*8, kernel_size, strides, padding='same', activation="relu")(x_sino)
    con4 =  Conv2D(base_filter*8, kernel_size, strides, padding='same', activation="relu")(con4)
    x_sino = MaxPooling2D(pool_size=(2, 2))(con4)
    
    con5 =  Conv2D(base_filter*16, kernel_size, strides, padding='same', activation="relu")(x_sino)
    con5 =  Conv2D(base_filter*16, kernel_size, strides, padding='same', activation="relu")(con5)


    # up
    up = Conv2DTranspose(base_filter*8, kernel_size, strides=(2,2), padding='same', activation='relu')(con5)
    merge1 = concatenate([up, con4], axis=-1)
    con6 =  Conv2D(base_filter*8, kernel_size, strides, padding='same', activation="relu")(merge1)
    con6 =  Conv2D(base_filter*8, kernel_size, strides, padding='same', activation="relu")(con6)
    up = Conv2DTranspose(base_filter*4, kernel_size, strides=(2,2), padding='same', activation='relu')(con6)
    merge2 = concatenate([up, con3], axis=-1)
    con7 =  Conv2D(base_filter*4, kernel_size, strides, padding='same', activation="relu")(merge2)
    con7 =  Conv2D(base_filter*4, kernel_size, strides, padding='same', activation="relu")(con7)
    up = Conv2DTranspose(base_filter*2, kernel_size, strides=(2,2), padding='same', activation='relu')(con7)
    merge3 = concatenate([up, con2], axis=-1)
    con8 =  Conv2D(base_filter*2, kernel_size, strides, padding='same', activation="relu")(merge3)
    con8 =  Conv2D(base_filter*2, kernel_size, strides, padding='same', activation="relu")(con8)
    up = Conv2DTranspose(base_filter, kernel_size, strides=(2,2), padding='same', activation='relu')(con8)
    merge4 = concatenate([up, con1], axis=-1)
    con9 =  Conv2D(base_filter, kernel_size, strides, padding='same', activation="relu")(merge4)
    con9 =  Conv2D(base_filter, kernel_size, strides, padding='same', activation="relu")(con9)
#     out = Conv2D(1, kernel_size=(1, 1), strides=(1,1), padding='same')(con9)
    out_sino = Conv2D(1, kernel_size=(1, 1), strides=(1,1), padding='same', activation="relu")(con9)
    
    # rescale 
    if irt_mode == 'classic':
        # rescale sino to max intensity
        out_sino = tf.keras.layers.Rescaling(max_sinogram_intensity, offset=0.0,)(out_sino)
        # get scan recon
        out_scan = InverseRadonTransform(name='irt')(out_sino)
#         out_scan = tf.random.uniform((362, 362, 1))
        # rescale scan to [0,1]
        out_scan = tf.keras.layers.Rescaling(1./max_scan_intensity, offset=0.0, name = 'out_scan')(out_scan)
        # rescale sino to [0,1]
        out_sino = tf.keras.layers.Rescaling(1./max_sinogram_intensity, offset=0.0, name = 'out_sino')(out_sino)
        
        # calculate metrics 
        out_sino, out_scan = MetricLayer(mse, ssim, psnr)([input_sino, input_scan, out_sino, out_scan])
        
        # calculate loss (just on sinograms)
        out_sino = LossLayer(mse)([input_sino, out_sino])

        model = tf.keras.Model(inputs=[input_sino_, input_scan_], 
                                   outputs=[out_sino, out_scan])

        return model
    
    if irt_mode == 'none':
        
        out_sino = MetricLayer(mse, ssim, psnr, scan_metrics=False)([input_sino, out_sino])
        
        # calculate loss (just on sinograms)
        out_sino = LossLayer(mse)([input_sino, out_sino])

        model = tf.keras.Model(inputs=[input_sino_, input_scan_], 
                                   outputs=out_sino)
        return model




#### DATALOADER

# These were calculated from our training set

def get_ds(dset, version=2, n=1, ds_bsz = 1, mod='sino', standardize=False):
    
    mean=56.23
    variance=3902.5
    sdev = np.sqrt(variance)
    
    class generator:
        def __call__(self, index):
            scan, sino, settings = load_sim_data(int(index), dset=dset, version=version)
            if version == 2:
                pass
            if version == 1:
                sino = resize(sino, (512,512)) # tshould this be in transform...
            sino = sino[:,:,np.newaxis]
            scan = scan[:,:,np.newaxis]
#             print(f"Sinogram shape: {sino.shape}, Scan shape: {scan.shape}")
            
            if standardize:
                sino -= mean
                sino  /= sdev

            yield (sino, scan), (sino, scan)

    # get the indices for the files that we want to use
    file_inds = np.arange(1,n+1)

    # Create dataset from generator in order to deliver tuple as output 
    # (which is necessary to have data and target as the same image)
    cycle_length = 1
    block_length = 2
    ds = tf.data.Dataset.from_tensor_slices(file_inds)
    ds = ds.interleave(lambda file_ind: tf.data.Dataset.from_generator(
            generator(), 
            output_signature=(
                (
                tf.TensorSpec(shape=(512, 512, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(362, 362, 1), dtype=tf.float32)
              ),
               (
                tf.TensorSpec(shape=(512, 512, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(362, 362, 1), dtype=tf.float32) 
            )
            ),
            args=(file_ind,)),
           cycle_length, block_length)
    ds = ds.batch(ds_bsz)
    
    return ds

# a = next(iter(get_ds('train', version=2, n=10, ds_bsz=2)))
def save_pickle(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def parse_option():

    # for more potential args: https://github.com/HobbitLong/SupContrast

    parser = argparse.ArgumentParser("argument for training")


    parser.add_argument(
        "--proj_name", type=str, default="testing", help="name for group of runs"
    )

    parser.add_argument(
        "--sr",
        type=float,
        default=0.1,
        help="sampling rate",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,  # I should enforce that len of this equals number of parts
        help="number of training epochs for each part",
    )

    parser.add_argument(
        "--bsz",  # I may want to make this different for contrastive learning
        type=int,
        default=4,
        help="batch size in each iteration",
    )

    parser.add_argument(
        "--lr",  # I may want to make this different for contrastive learning
        type=int,
        default=0.001,
        help="batch size in each iteration",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="num workers. 0 means data will be loaded in the main process",
    )

    parser.add_argument(
        "--n_train",
        type=int,
        default=100,
        help="n train",
    )

    parser.add_argument(
        "--n_val",
        type=int,
        default=25,
        help="n val",
    )

    parser.add_argument(
        "--irt_mode",
        type=str,
        default='none',
        help="irt mode",
    )

    parser.add_argument(
        "--version",
        type=int,
        default=2,
        help="version of sim data",
    )

    # parser.add_argument(
    #     "--save_freq",
    #     type=int,
    #     default=5,
    #     help="save frequency. default is set high so it won't work",
    # )

    # parser.add_argument(
    #     "--random_seed",
    #     type=int,
    #     default=31,
    #     help="random seed for data splitting. try not to change",
    # )
    opt = parser.parse_args()

    return opt




def run_model(sr = 0.1, epochs = 1, version = 2, ds_bsz = 4, n_train = 100, n_val = 20, irt_mode = 'classic'):


    unet = unet_model(sr=sr,
                    irt_mode= irt_mode)

    unet.compile(optimizer=Adam(learning_rate=0.001),  # pick an optimizer
                # loss and metrics already declared
                        )  

    train_ds = get_ds('train', version, n=n_train, ds_bsz=ds_bsz)
    val_ds = get_ds('val', version, n =n_val, ds_bsz=ds_bsz)
    hist = unet.fit(
                train_ds, # data and targs
                epochs=epochs,
                validation_data= val_ds,
                workers = 4
    )

    return hist

def main():

    opt = parse_option()

    # make results folder if it doesn't exist
    root = Path("/hpc/group/tdunn/bel25/RACTS/results")
    exp_str = f"epochs-{opt.epochs}_version-{opt.version}_bsz-{opt.bsz}_ntrain-{opt.n_train}_n_val-{opt.n_val}_irt_mode-{opt.irt_mode}"
    results_folder = root  / opt.proj_name / exp_str
    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)

    unet = unet_model(sr=opt.sr,
                    irt_mode= opt.irt_mode)

    unet.compile(optimizer=Adam(learning_rate=opt.lr),  # pick an optimizer
                # loss and metrics already declared
                        )  

    train_ds = get_ds('train', opt.version, n=opt.n_train, ds_bsz=opt.bsz)
    val_ds = get_ds('val', opt.version, n = opt.n_val, ds_bsz = opt.bsz)
    hist = unet.fit(
                train_ds, # data and targs
                epochs=opt.epochs,
                validation_data = val_ds,
                workers = opt.workers
    )


    # Save the weights
    model_path = results_folder / f"MODEL_sr-{opt.sr}"
    unet.save(model_path)
    # unet.save_weights(weights_path)

    # save images
    hdf5_file_path = results_folder / f"IMGS_sr-{opt.sr}.h5"
    save_run_h5(unet, hdf5_file_path, n = 5)

    # save metrics
    hist_path = results_folder / f"HIST_sr-{opt.sr}.pkl"
    save_pickle(hist.history, hist_path)


if __name__ == "__main__":
    main()

