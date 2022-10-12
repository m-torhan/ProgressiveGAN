import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, MaxPool2D, Input, Dense, Flatten, Dropout, Concatenate, Layer, LeakyReLU, Reshape, AveragePooling2D, Add

import numpy as np

from time import perf_counter
import traceback
from functools import partial
import json

from losses import *
from custom_layers import *
from utils import *

class ProgressiveGAN(object):
    __latent_dim                : int
    __initial_image_size        : int
    __final_image_size          : int
    __image_channels            : int
    __gan_optimizer             : (str | keras.optimizers.Optimizer)
    __discriminator_optimizer   : (str | keras.optimizers.Optimizer)
    
    __steps                     : list[int]

    __generator                 : keras.Model
    __discriminator             : keras.Model

    __gan                       : keras.Model

    def __init__(self, latent_dim : int =128, initial_image_size : int =4, final_image_size : int =512, image_channels : int =3,
                 gan_optimizer : (str | keras.optimizers.Optimizer) ='adam', discriminator_optimizer : (str | keras.optimizers.Optimizer) ='adam'):
        self.__latent_dim = latent_dim
        self.__initial_image_size = initial_image_size
        self.__final_image_size = final_image_size
        self.__image_channels = image_channels
        self.__gan_optimizer = gan_optimizer
        self.__discriminator_optimizer = discriminator_optimizer
        
        self.__checkpoint_idx = 0

        self.__steps = []
        
        image_size = initial_image_size
        while image_size <= final_image_size:
            self.__steps.append(image_size)
            image_size <<= 1
        
        self.__total_epochs = 0

        self.__generator = []
        self.__discriminator = []
        self.__discriminator_gp = []
        self.__gan = []

        self.__init_generator()
        self.__init_discriminator()
        self.__init_gan()
    
        self.__timer = perf_counter()
    
    @property
    def generator(self):
        return self.__generator

    @property
    def discriminator(self):
        return self.__discriminator

    @property
    def gan(self):
        return self.__gan

    def sample_latent_space(self, n : int) -> np.ndarray:
        # sample from unit hypersphere
        normal_sample = np.random.normal(size=(n, 1, 1, self.__latent_dim))
        
        return normal_sample/np.sqrt((normal_sample**2).sum(axis=3))[:,:,:,np.newaxis]
    
    def __train_models(self, step : int, fade : bool, image_generator : ImageGenerator, batches_per_step : int =32, discriminator_train_per_gan_train : int =5,
                       checkpoint_save_interval=1000, tensorboard_callback=None, fit_checkpoint : FitCheckpoint =None):
        generator           = self.__generator[step][int(fade)]
        discriminator       = self.__discriminator_gp[step][int(fade)]
        gan                 = self.__gan[step][int(fade)]
        
        g_loss_total = .0
        d_loss_total = .0
        d_loss_generated_total = .0
        d_loss_real_total = .0
        d_loss_gradient_penalty_total = .0
        
        epoch_offset = 0
        
        if fit_checkpoint is not None:
            g_loss_total                    = fit_checkpoint.fit_progress['g_loss_total']
            d_loss_total                    = fit_checkpoint.fit_progress['d_loss_total']
            d_loss_generated_total          = fit_checkpoint.fit_progress['d_loss_generated_total']
            d_loss_real_total               = fit_checkpoint.fit_progress['d_loss_real_total']
            d_loss_gradient_penalty_total   = fit_checkpoint.fit_progress['d_loss_gradient_penalty_total']
            epoch_offset                    = fit_checkpoint.fit_progress['epoch']
        
        image_generator.set_fade(fade)
        
        for epoch in range(epoch_offset, batches_per_step + epoch_offset):
            # adjust fade in parameter
            alpha = 0
            if fade:
                alpha = epoch/batches_per_step
                for model in (generator, discriminator, gan):
                    for layer in model.layers:
                        if isinstance(layer, WeightedSum):
                            K.set_value(layer.alpha, alpha)
                
                image_generator.set_fade_alpha(alpha)
                
            # train discriminator
            d_loss_generated = 0.
            d_loss_real = 0.
            d_loss_gradient_penalty = 0.
            
            for _ in range(discriminator_train_per_gan_train):
                latent_noise = self.sample_latent_space(image_generator.batch_size)

                # generated_images = generator.predict(latent_noise)
                real_images = image_generator.get_batch()
                
                generated_labels = -1. * np.ones((image_generator.batch_size, 1))
                real_labels = np.ones((image_generator.batch_size, 1))
                dummy_labels = np.zeros((image_generator.batch_size, 1))

                # combined_images = np.concatenate([generated_images, real_images])

                # labels = np.ones((batch_size << 1, 1))
                # labels[:batch_size,] = 0

                generated_labels += .1 * np.random.normal(0, 1, generated_labels.shape)
                real_labels += .1 * np.random.normal(0, 1, real_labels.shape)
                
                loss = discriminator.train_on_batch([real_images, latent_noise], [real_labels, generated_labels, dummy_labels])
                d_loss_real += loss[0]
                d_loss_generated += loss[1]
                d_loss_gradient_penalty += loss[2]
            
            d_loss_generated /= discriminator_train_per_gan_train
            d_loss_real /= discriminator_train_per_gan_train
            d_loss_gradient_penalty /= discriminator_train_per_gan_train
            
            d_loss = (d_loss_generated + d_loss_real + d_loss_gradient_penalty)/2
            
            # train generator
            latent_noise = self.sample_latent_space(image_generator.batch_size)

            misleading_labels = np.ones((image_generator.batch_size, 1))
            misleading_labels += .1 * np.random.normal(0, 1, misleading_labels.shape)

            g_loss = gan.train_on_batch(latent_noise, misleading_labels)
            
            g_loss_total                    += g_loss
            d_loss_total                    += d_loss
            d_loss_generated_total          += d_loss_generated
            d_loss_real_total               += d_loss_real
            d_loss_gradient_penalty_total   += d_loss_gradient_penalty

            if epoch + 1 < batches_per_step:
                self.__print_fit_progress(self.__steps[step], step, fade, epoch + 1, batches_per_step + epoch_offset,
                                          alpha,
                                          g_loss,
                                          d_loss,
                                          d_loss_generated, d_loss_real, d_loss_gradient_penalty)
            else:
                self.__print_fit_progress(self.__steps[step], step, fade, epoch + 1, batches_per_step + epoch_offset,
                                          alpha,
                                          g_loss_total/batches_per_step,
                                          d_loss_total/batches_per_step,
                                          d_loss_generated_total/batches_per_step, d_loss_real_total/batches_per_step, d_loss_gradient_penalty_total/batches_per_step)

            if (epoch + 1) % checkpoint_save_interval == 0:
                fit_checkpoint = FitCheckpoint(
                    {'batches_per_step': batches_per_step,
                     'discriminator_train_per_gan_train': discriminator_train_per_gan_train,
                     'checkpoint_save_interval': checkpoint_save_interval },
                    {'step': step,
                     'fade': fade,
                     'epoch': epoch,
                     'g_loss_total': g_loss_total,
                     'd_loss_total': d_loss_total,
                     'd_loss_generated_total': d_loss_generated_total,
                     'd_loss_real_total': d_loss_real_total,
                     'd_loss_gradient_penalty_total': d_loss_gradient_penalty_total})

                fit_checkpoint.save(f'./fit_checkpoints/checkpoint_{self.__checkpoint_idx}')
                self.save(f'./model/model_checkpoint_{self.__checkpoint_idx}')
                
                self.__checkpoint_idx ^= 1
            
            if tensorboard_callback is not None:
                tensorboard_callback.on_batch_end(
                    epoch, step, fade,
                    {'loss': {      #'d_loss_generated' : d_loss_generated,
                                    #'d_loss_real' : d_loss_real,
                                    'd_loss' : d_loss,
                                    'g_loss': g_loss}})
            
            self.__total_epochs += 1
        
        tensorboard_callback.on_epoch_end(step, fade)
        
        # self.__discriminator_optimizer.learning_rate.assign(self.__discriminator_optimizer.learning_rate * .8)
        # if self.__discriminator_optimizer != self.__gan_optimizer:
        #     self.__gan_optimizer.learning_rate.assign(self.__gan_optimizer.learning_rate * .8)
            
    def fit(self, image_generator : ImageGenerator, batches_per_step : int =32, discriminator_train_per_gan_train=5, checkpoint_save_interval=1000, tensorboard_callback=None):
        try:
            image_generator.set_images_size(self.__steps[0])
            
            self.__print_fit_progress_header()
            self.__train_models(step=0, fade=False, image_generator=image_generator, batches_per_step=batches_per_step,
                                discriminator_train_per_gan_train=discriminator_train_per_gan_train,
                                checkpoint_save_interval=checkpoint_save_interval, tensorboard_callback=tensorboard_callback)

            for step in range(1, len(self.__steps)):
                img_size = self.__steps[step]
                
                image_generator.set_images_size(img_size)
                
                self.__train_models(step=step, fade=True, image_generator=image_generator, batches_per_step=batches_per_step,
                                    discriminator_train_per_gan_train=discriminator_train_per_gan_train,
                                    checkpoint_save_interval=checkpoint_save_interval, tensorboard_callback=tensorboard_callback)
                
                self.__train_models(step=step, fade=False, image_generator=image_generator, batches_per_step=batches_per_step,
                                    discriminator_train_per_gan_train=discriminator_train_per_gan_train,
                                    checkpoint_save_interval=checkpoint_save_interval, tensorboard_callback=tensorboard_callback)
                
        except:
            traceback.print_exc()
        
        if tensorboard_callback is not None:
            tensorboard_callback.on_fit_end()
    
    def resume_fit_from_checkpoint(self, fit_checkpoint: FitCheckpoint, image_generator : ImageGenerator, tensorboard_callback=None):
        try:
            checkpoint_step                     = fit_checkpoint.fit_progress['step']
            checkpoint_fade                     = fit_checkpoint.fit_progress['fade']
            checkpoint_epoch                    = fit_checkpoint.fit_progress['epoch']
            
            batches_per_step                    = fit_checkpoint.fit_params['batches_per_step']
            discriminator_train_per_gan_train   = fit_checkpoint.fit_params['discriminator_train_per_gan_train']
            checkpoint_save_interval            = fit_checkpoint.fit_params['checkpoint_save_interval']
            
            image_generator.set_images_size(self.__steps[checkpoint_step])
            
            self.__print_fit_progress_header()
            
            if checkpoint_fade:
                # resume on fade step
                self.__train_models(step=checkpoint_step, fade=True, image_generator=image_generator, batches_per_step=(batches_per_step - checkpoint_epoch),
                                    discriminator_train_per_gan_train=discriminator_train_per_gan_train,
                                    checkpoint_save_interval=checkpoint_save_interval, tensorboard_callback=tensorboard_callback,
                                    fit_checkpoint=fit_checkpoint)
                
                self.__train_models(step=checkpoint_step, fade=False, image_generator=image_generator, batches_per_step=batches_per_step,
                                    discriminator_train_per_gan_train=discriminator_train_per_gan_train,
                                    checkpoint_save_interval=checkpoint_save_interval, tensorboard_callback=tensorboard_callback)
            else:
                # resume on non fade step
                self.__train_models(step=checkpoint_step, fade=False, image_generator=image_generator, batches_per_step=(batches_per_step - checkpoint_epoch),
                                    discriminator_train_per_gan_train=discriminator_train_per_gan_train,
                                    checkpoint_save_interval=checkpoint_save_interval, tensorboard_callback=tensorboard_callback,
                                    fit_checkpoint=fit_checkpoint)

            # proceed to next steps
            for step in range(checkpoint_step + 1, len(self.__steps)):
                img_size = self.__steps[step]
                
                image_generator.set_images_size(img_size)
                
                self.__train_models(step=step, fade=True, image_generator=image_generator, batches_per_step=batches_per_step,
                                    discriminator_train_per_gan_train=discriminator_train_per_gan_train,
                                    checkpoint_save_interval=checkpoint_save_interval, tensorboard_callback=tensorboard_callback)
                
                self.__train_models(step=step, fade=False, image_generator=image_generator, batches_per_step=batches_per_step,
                                    discriminator_train_per_gan_train=discriminator_train_per_gan_train,
                                    checkpoint_save_interval=checkpoint_save_interval, tensorboard_callback=tensorboard_callback)
                
        except:
            traceback.print_exc()
        
        if tensorboard_callback is not None:
            tensorboard_callback.on_fit_end()
    
    def save(self, path : str):
        if not os.path.isdir(path):
            os.mkdir(path)
        
        model_params = {
            'latent_dim'        : self.__latent_dim,
            'initial_image_size': self.__initial_image_size,
            'final_image_size'  : self.__final_image_size,
            'image_channels'    : self.__image_channels
        }
        
        with open(os.path.join(path, 'model_params.json'), 'w') as model_params_file:
            json.dump(model_params, model_params_file)
        
        for step, _ in enumerate(self.__steps):
            for fade in (0, 1):
                self.__generator[step][int(fade)].save_weights(os.path.join(path, f'generator_{step}_{int(fade)}.h5'))
                self.__discriminator[step][int(fade)].save_weights(os.path.join(path, f'discriminator_{step}_{int(fade)}.h5'))
                self.__discriminator_gp[step][int(fade)].save_weights(os.path.join(path, f'discriminator_gp_{step}_{int(fade)}.h5'))
                self.__gan[step][int(fade)].save_weights(os.path.join(path, f'gan_{step}_{int(fade)}.h5'))
    
    @classmethod
    def load(self, path : str, gan_optimizer : (str | keras.optimizers.Optimizer) ='adam', discriminator_optimizer : (str | keras.optimizers.Optimizer) ='adam'):
        with open(os.path.join(path, 'model_params.json'), 'r') as model_params_file:
            model_params = json.load(model_params_file)
        
        progan = ProgressiveGAN(model_params['latent_dim'], model_params['initial_image_size'], model_params['final_image_size'], model_params['image_channels'],
                                gan_optimizer, discriminator_optimizer)
            
        for step, _ in enumerate(progan.__steps):
            for fade in (0, 1):
                progan.__generator[step][int(fade)].load_weights(os.path.join(path, f'generator_{step}_{int(fade)}.h5'))
                progan.__discriminator[step][int(fade)].load_weights(os.path.join(path, f'discriminator_{step}_{int(fade)}.h5'))
                progan.__discriminator_gp[step][int(fade)].load_weights(os.path.join(path, f'discriminator_gp_{step}_{int(fade)}.h5'))
                progan.__gan[step][int(fade)].load_weights(os.path.join(path, f'gan_{step}_{int(fade)}.h5'))
        
        return progan
    
    def __set_trainable(self, model, value):
        model.trainable = value
        
        for layer in model.layers:
            layer.trainable = value
    
    def __print_fit_progress_header(self):
        print('| image size       | step | fade | epoch            | time     | g_loss                           | d_loss                           | d_loss_generated                 | d_loss_real                      | d_loss_gradient_penalty          |', end='')
    
    def __print_fit_progress(self, img_size, step, fade, epoch, total_epochs, alpha, g_loss, d_loss, d_loss_generated, d_loss_real, d_loss_gradient_penalty):
        if epoch == 1:
            self.__timer = perf_counter()
            print()
            
        epoch_time = perf_counter() - self.__timer
        
        time = 0
        if epoch == total_epochs:
            # time passed
            time = epoch_time
        else:
            # eta
            time = epoch_time*(total_epochs - epoch)/epoch
        
        time_str = ''
        if time < 60:
            time_str = f'{int(time)}.{int(time*100)%100:02d}'
        elif time < 60*60:
            time_str = f'{(int(time)//60)%60}:{int(time)%60:02d}'
        else:
            time_str = f'{(int(time)//(60*60))%60}:{(int(time)//60)%60:02d}:{int(time)%60:02d}'
        
        img_size_str = ''
        
        if alpha > 0:
            img_size = f'{alpha:.2f} '
            
        if fade:
            img_size_str += f'{img_size//2} -> {img_size}'
        else:
            img_size_str += f'{img_size}'
        
        epoch_str = f'{epoch} / {total_epochs}'
    
        print(f'\r| {img_size_str:>16s} | {step:4d} | {int(fade):4d} | {epoch_str:>16s} | {time_str:>8s} | {g_loss:32f} | {d_loss:32f} | {d_loss_generated:32f} | {d_loss_real:32f} | {d_loss_gradient_penalty:32f} | ', end='')

    def __init_generator(self):
        kernel_initializer = keras.initializers.HeNormal()
        kernel_constraint = None # keras.constraints.MaxNorm(1.)
        
        self.__generator = []
        
        generator_input = x = Input((1, 1, self.__latent_dim))
        
        x = Conv2DTranspose(self.__latent_dim, 4, kernel_initializer=kernel_initializer, kernel_constraint=kernel_constraint)(x)
        x = PixelNormalization()(x)
        x = LeakyReLU(alpha=.2)(x)

        x = Conv2D(self.__latent_dim, 3, padding='same', kernel_initializer=kernel_initializer, kernel_constraint=kernel_constraint)(x)
        x = PixelNormalization()(x)
        x = LeakyReLU(alpha=.2)(x)

        output_size = 4

        while output_size < self.__initial_image_size:
            filters = self.__filters_count(output_size)

            x = UpSampling2D()(x)
            
            x = Conv2DTranspose(filters, 3, padding='same', kernel_initializer=kernel_initializer, kernel_constraint=kernel_constraint)(x)
            x = PixelNormalization()(x)
            x = LeakyReLU(alpha=.2)(x)
            
            x = Conv2D(filters, 3, padding='same', kernel_initializer=kernel_initializer, kernel_constraint=kernel_constraint)(x)
            x = PixelNormalization()(x)
            x = LeakyReLU(alpha=.2)(x)

            output_size <<= 1

        x = Conv2D(self.__image_channels, 1, padding='same', kernel_initializer=kernel_initializer, kernel_constraint=kernel_constraint)(x)
        
        generator = keras.Model(generator_input, x, name=f'generator_{self.__initial_image_size:}x{self.__initial_image_size:}')

        self.__generator.append([generator, generator])
        
        for _ in range(1, len(self.__steps)):
            next_generators = self.__add_generator_block(self.__generator[-1][0])
            self.__generator.append(next_generators)
    
    def __add_generator_block(self, generator : keras.Model):
        kernel_initializer = keras.initializers.HeNormal()
        kernel_constraint = None # keras.constraints.MaxNorm(1.)
        
        prev_block_end = generator.layers[-2].output
        
        upsampling = x = UpSampling2D()(prev_block_end)
        
        output_image_size = x.shape[1]
        filters = self.__filters_count(output_image_size)
        
        x = Conv2D(filters, 3, padding='same', kernel_initializer=kernel_initializer, kernel_constraint=kernel_constraint)(x)
        x = PixelNormalization()(x)
        x = LeakyReLU(alpha=.2)(x)

        x = Conv2D(filters, 3, padding='same', kernel_initializer=kernel_initializer, kernel_constraint=kernel_constraint)(x)
        x = PixelNormalization()(x)
        x = LeakyReLU(alpha=.2)(x)
        
        new_generator_output = Conv2D(self.__image_channels, 1, padding='same', kernel_initializer=kernel_initializer, kernel_constraint=kernel_constraint)(x)
        
        new_generator = keras.Model(generator.input, new_generator_output, name=f'generator_{output_image_size}x{output_image_size}')
        
        generator_output = generator.layers[-1]
        generator_output_upscaled = generator_output(upsampling)
        
        combined = WeightedSum()((new_generator_output, generator_output_upscaled))
        
        new_generator_fade = keras.Model(generator.input, combined, name=f'generator_fade_{output_image_size}x{output_image_size}')
        
        return [new_generator, new_generator_fade]

    def __add_gradient_penalty_to_discriminator(self, discriminator, generator, img_size):
        self.__set_trainable(generator, False)
        
        latent_input = Input((1, 1, self.__latent_dim))
        
        real_img = Input((img_size, img_size, self.__image_channels))
        fake_img = generator(latent_input)
        
        real_predict = discriminator(real_img)
        fake_predict = discriminator(fake_img)
        
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        
        interpolated_predict = discriminator(interpolated_img)
        
        partial_gp_loss = partial(gradient_penalty_loss, interpolated_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty_loss'
        
        discriminator_gp = keras.Model(inputs=[real_img, latent_input], outputs=[real_predict, fake_predict, interpolated_predict],
                                       name=f'discriminator_gp_{img_size}x{img_size}')
        
        discriminator_gp.compile(optimizer=self.__discriminator_optimizer,
                                 loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss],
                                 loss_weights=[1, 1, 10])
        
        self.__set_trainable(generator, True)
        
        return discriminator_gp

    def __init_discriminator(self):
        kernel_initializer = keras.initializers.HeNormal()
        kernel_constraint = None #  keras.constraints.MaxNorm(1.)
        
        self.__discriminator = []
        self.__discriminator_gp = []
        
        discriminator_input = x = Input((self.__initial_image_size, self.__initial_image_size, self.__image_channels))

        x = Conv2D(self.__filters_count(self.__initial_image_size << 1), 1, padding='same', kernel_initializer=kernel_initializer, kernel_constraint=kernel_constraint)(x)
        x = LeakyReLU(alpha=.2)(x)

        output_size = self.__initial_image_size

        while output_size > 4:
            x = Conv2D(self.__filters_count(output_size << 1), 3, padding='same', kernel_initializer=kernel_initializer, kernel_constraint=kernel_constraint)(x)
            x = LeakyReLU(alpha=.2)(x)
            
            x = Conv2D(self.__filters_count(output_size), 3, padding='same', kernel_initializer=kernel_initializer, kernel_constraint=kernel_constraint)(x)
            x = LeakyReLU(alpha=.2)(x)
            
            x = AveragePooling2D()(x)

            output_size >>= 1

        x = MinibatchStdev()(x)
        
        x = Conv2D(self.__filters_count(output_size), 3, padding='same', kernel_initializer=kernel_initializer, kernel_constraint=kernel_constraint)(x)
        x = LeakyReLU(alpha=.2)(x)
        
        x = Conv2D(self.__latent_dim, 4, kernel_initializer=kernel_initializer, kernel_constraint=kernel_constraint)(x)
        x = LeakyReLU(alpha=.2)(x)
        
        x = Flatten()(x)
        
        x = Dense(1)(x)

        discriminator = keras.Model(discriminator_input, x,
                                    name=f'discriminator_{self.__initial_image_size}x{self.__initial_image_size}')
        
        discriminator_gp = self.__add_gradient_penalty_to_discriminator(discriminator, self.__generator[0][0], self.__initial_image_size)
        
        self.__discriminator.append([discriminator, discriminator])
        self.__discriminator_gp.append([discriminator_gp, discriminator_gp])

        for i in range(1, len(self.__steps)):
            next_discriminators, next_discriminators_gp = self.__add_discriminator_block(i, self.__discriminator[-1][0])
            self.__discriminator.append(next_discriminators)
            self.__discriminator_gp.append(next_discriminators_gp)
    
    def __add_discriminator_block(self, index, discriminator : keras.Model):
        kernel_initializer = keras.initializers.HeNormal()
        kernel_constraint = None # keras.constraints.MaxNorm(1.)
        
        discriminator_input_shape = discriminator.input.shape
        new_discriminator_input_shape = (discriminator_input_shape[-3] << 1, discriminator_input_shape[-2] << 1, discriminator_input_shape[-1])
        output_image_size = new_discriminator_input_shape[0]
        
        new_discriminator_input = x = Input(shape=new_discriminator_input_shape)
        
        x = Conv2D(self.__filters_count(output_image_size << 1), 1, padding='same', kernel_initializer=kernel_initializer, kernel_constraint=kernel_constraint)(x)
        x = LeakyReLU(alpha=.2)(x)
        
        x = Conv2D(self.__filters_count(output_image_size << 1), 3, padding='same', kernel_initializer=kernel_initializer, kernel_constraint=kernel_constraint)(x)
        x = LeakyReLU(alpha=.2)(x)
        
        x = Conv2D(self.__filters_count(output_image_size), 3, padding='same', kernel_initializer=kernel_initializer, kernel_constraint=kernel_constraint)(x)
        x = LeakyReLU(alpha=.2)(x)
        
        new_block_end = x = AveragePooling2D()(x)
        
        # skip the input, 1x1 and activation layers of the old model
        for i in range(3, len(discriminator.layers)):
            x = discriminator.layers[i](x)
            
        new_discriminator = keras.Model(new_discriminator_input, x, name=f'discriminator_{output_image_size}x{output_image_size}')
        
        new_discriminator_gp = self.__add_gradient_penalty_to_discriminator(new_discriminator, self.__generator[index][0], output_image_size)
        
        x = AveragePooling2D()(new_discriminator_input)
        x = discriminator.layers[1](x)  # 1x1 conv
        x = discriminator.layers[2](x)  # activation
        
        x = WeightedSum()([new_block_end, x])
        
        # same as above
        for i in range(3, len(discriminator.layers)):
            x = discriminator.layers[i](x)
            
        new_discriminator_fade = keras.Model(new_discriminator_input, x, name=f'discriminator_fade_{output_image_size}x{output_image_size}')
        
        new_discriminator_gp_fade = self.__add_gradient_penalty_to_discriminator(new_discriminator_fade, self.__generator[index][1], output_image_size)
        
        return [new_discriminator, new_discriminator_fade], [new_discriminator_gp, new_discriminator_gp_fade]
    
    def __init_gan(self):
        self.__gan = []
        
        for generators, discriminators in zip(self.__generator, self.__discriminator):
            # straight-through model
            self.__set_trainable(discriminators[0], False)
            self.__set_trainable(generators[0], True)
            
            gan = keras.Sequential(name=f'gan_{generators[0].output.shape[1]}x{generators[0].output.shape[1]}')
            gan.add(generators[0])
            gan.add(discriminators[0])
            
            gan.compile(loss=wasserstein_loss, optimizer=self.__gan_optimizer)
            
            # fade-in model
            self.__set_trainable(discriminators[1], False)
            self.__set_trainable(generators[1], True)
            
            gan_fade = keras.Sequential(name=f'gan_fade_{generators[0].output.shape[1]}x{generators[0].output.shape[1]}')
            gan_fade.add(generators[1])
            gan_fade.add(discriminators[1])
            
            gan_fade.compile(loss=wasserstein_loss, optimizer=self.__gan_optimizer)
            
            self.__gan.append((gan, gan_fade))
            
    def __filters_count(self, output_size):
        filters = self.__latent_dim
        while output_size*filters >= self.__final_image_size*16:
            filters //= 2
        
        return filters