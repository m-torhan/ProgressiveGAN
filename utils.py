import tensorflow as tf

import numpy as np

import cv2

import os
import datetime
from time import sleep
import threading

from random import sample

class ImageGenerator(object):
    def __init__(self, images_folder_path, initial_images_size=4, batch_size=32, image_channels=3):
        self.__images_folder_path = images_folder_path
        self.__images_size = initial_images_size
        self.__batch_size = batch_size
        self.__image_channels = image_channels
        
        self.__filenames = []
        
        self.__cached_bank = 0
        self.__cached_batch = [None, None]
        self.__cached_size = [0, 0]
        self.__cached_ready = [True, True]

        for _, _, fnames in os.walk(self.__images_folder_path):
            for fname in fnames:
                if fname.split('.')[-1] in ('jpg', 'jpeg'):
                    self.__filenames.append(fname)
            break
    
        print(f'Loaded {len(self.__filenames)} images.')
    
    @property
    def batch_size(self):
        return self.__batch_size
    
    def set_images_size(self, size):
        self.__images_size = size

    def get_batch(self):
        while not self.__cached_ready[self.__cached_bank]:
            sleep(.01)
        
        result = self.__cached_batch[self.__cached_bank]
        result_size = self.__cached_size[self.__cached_bank]
        
        self.__cached_bank ^= 1
        self.__cached_ready[self.__cached_bank] = False
        prepare_thread = threading.Thread(target=self.__prepare_cached_batch)
        prepare_thread.start()
        
        if result_size != self.__images_size:
            # wrong size
            return self.get_batch()
        
        return result
        

    def __prepare_cached_batch(self):
        img_size = self.__images_size
        result = np.zeros((self.__batch_size, img_size, img_size, self.__image_channels))

        fnames = sample(self.__filenames, self.__batch_size)

        for i in range(self.__batch_size):
            img = cv2.imread(os.path.join(self.__images_folder_path, fnames[i]))[:,:,::-1]
            min_size = min(img.shape[:2])
            img = img[(img.shape[0] - min_size)//2:(img.shape[0] + min_size)//2,
                      (img.shape[1] - min_size)//2:(img.shape[1] + min_size)//2]
            img = cv2.resize(img, (img_size,)*2).astype(np.float32)
            img -= img.min()
            img /= (img.max() + 1e-9)
            # [0, 1] -> [-1, 1]
            img *= 2.
            img -= 1.
            if len(img.shape) == 2:
                img = img[:,:,np.newaxis]
            result[i,] = img[:,:,:self.__image_channels]
        
        self.__cached_batch[self.__cached_bank] = result
        self.__cached_size[self.__cached_bank] = img_size
        self.__cached_ready[self.__cached_bank] = True

class TensorBoardCallback(object):
    def __init__(self, logdir : str, model = None,
                 tensorboard_metrics_save_interval : int =20, tensorboard_generator_preview_save_interval : int =100,
                 image_generator_preview_save_interval : int = 1000, frame_generator_preview_save_interval : int = 100,
                 use_tensorboard : bool =True):
        self.__datetime_str = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        try:
            os.mkdir(os.path.join('gen', self.__datetime_str))
        except:
            pass
        
        self.__logdir = os.path.join(logdir, self.__datetime_str)
        self.__model = model
        self.__tensorboard_metrics_save_interval = tensorboard_metrics_save_interval
        self.__tensorboard_generator_preview_save_interval = tensorboard_generator_preview_save_interval
        self.__image_generator_preview_save_interval = image_generator_preview_save_interval
        self.__frame_generator_preview_save_interval = frame_generator_preview_save_interval
        self.__use_tensorboard = use_tensorboard
        
        self.__generator_preview_vid_latent = model.sample_latent_space(1)
        self.__generator_preview_vid = cv2.VideoWriter(os.path.join('./gen', self.__datetime_str, 'generator_preview.mp4'), 0x7634706d, 60.0, (512, 512))
        
        self.__total_epochs = 0
        
        self.__writers = {}
        
        self.__metrics_interval = {}
        self.__metrics_interval_count = 0

        if model is not None:
            self.__preview_latent_noise = model.sample_latent_space(4)
    
    def on_epoch_end(self, epoch : int, step : int, fade : bool, metrics_dict : dict):
        self.__total_epochs += 1
        
        if self.__use_tensorboard:
            for metric_name, metric_dict in metrics_dict.items():
                if metric_name not in self.__metrics_interval.keys():
                    self.__metrics_interval[metric_name] = {}
                    
                for metric_subname, metric_value in metric_dict.items():
                    if metric_subname not in self.__metrics_interval[metric_name].keys():
                        self.__metrics_interval[metric_name][metric_subname] = .0
                
                    self.__metrics_interval[metric_name][metric_subname] += metric_value
                
            self.__metrics_interval_count += 1
            
            if self.__total_epochs % self.__tensorboard_metrics_save_interval == 0:
                self.__write_metrics()
                self.__metrics_interval_count = 0
                self.__metrics_interval = {}
            
            if self.__total_epochs % self.__tensorboard_generator_preview_save_interval == 0:
                self.__write_generator_preview(step, fade)
        
        frame = None
        if self.__total_epochs % self.__image_generator_preview_save_interval == 0 or\
           self.__total_epochs % self.__frame_generator_preview_save_interval == 0:
            frame = self.__model.generator[step][int(fade)].predict(self.__generator_preview_vid_latent)[0]
            # [-1., 1.] -> [0, 255]
            frame = np.clip((frame + 1.)*127.5, 0, 255).astype(np.uint8)
            frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_NEAREST)
            if len(frame.shape) == 2:
                frame = frame[:,:,np.newaxis]
            if frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                
        if self.__total_epochs % self.__frame_generator_preview_save_interval == 0:
            self.__generator_preview_vid.write(frame[:,:,::-1])
            
        if self.__total_epochs % self.__image_generator_preview_save_interval == 0:
            cv2.imwrite(os.path.join('./gen', self.__datetime_str , f'{2*step - int(fade):03d}_{epoch:06d}.jpg'), frame[:,:,::-1])
        
    def on_fit_end(self):
        self.__generator_preview_vid.release()
        
        for _, writer in self.__writers.items():
            writer.close()
        
        self.__writers = {}
        
    def __write_metrics(self):
        for metric_name, metric_dict in self.__metrics_interval.items():
            for loss_name, loss_value in metric_dict.items():
                if os.path.join(self.__logdir, loss_name) not in self.__writers.keys():
                    writer = tf.summary.create_file_writer(os.path.join(self.__logdir, loss_name))
                    self.__writers[os.path.join(self.__logdir, loss_name)] = writer
                    
                with self.__writers[os.path.join(self.__logdir, loss_name)].as_default():
                    tf.summary.scalar(metric_name, loss_value/self.__metrics_interval_count, step=self.__total_epochs)
    
    def __write_generator_preview(self, step : int, fade : bool):
        if os.path.join(self.__logdir, 'model') not in self.__writers.keys():
            writer = tf.summary.create_file_writer(os.path.join(self.__logdir, 'model'))
            self.__writers[os.path.join(self.__logdir, 'model')] = writer
            
        with self.__writers[os.path.join(self.__logdir, 'model')].as_default():
            preview_generated_images = self.__model.generator[step][fade].predict(self.__preview_latent_noise)
            tf.summary.image('Generator preview', preview_generated_images[:4,], step=self.__total_epochs, max_outputs=4)
    
    def __del__(self):
        for _, writer in self.__writers.items():
            writer.close()
    