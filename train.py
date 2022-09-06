import time
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from model import evaluate
from model import srgan



class Trainer:
    def __init__(self,
                 model,
                 loss,
                 learning_rate,
                 checkpoint_dir='./ckpt/edsr'):

        self.now = None
        self.loss = loss
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              psnr=tf.Variable(-1.0),
                                              optimizer=Adam(learning_rate),
                                              model=model)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                             directory=checkpoint_dir,
                                                             max_to_keep=3)

        self.restore()

    @property
    def model(self):
        return self.checkpoint.model

    def train(self, train_dataset, valid_dataset, steps, evaluate_every=1000, save_best_only=False):
        loss_mean = Mean()

        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint

        self.now = time.perf_counter()

        for lr, hr in train_dataset.take(steps - ckpt.step.numpy()):
            ckpt.step.assign_add(1)
            step = ckpt.step.numpy()

            loss = self.train_step(lr, hr)
            loss_mean(loss)

            if step % evaluate_every == 0:
                loss_value = loss_mean.result()
                loss_mean.reset_states()

                # Compute PSNR on validation dataset
                psnr_value, ssim_value = self.evaluate(valid_dataset)

                duration = time.perf_counter() - self.now
                print(f'{step}/{steps}: loss = {loss_value.numpy():.3f}, PSNR = {psnr_value.numpy():3f}, SSIM = {ssim_value.numpy():3f} ({duration:.2f}s)')

                if save_best_only and psnr_value <= ckpt.psnr:
                    self.now = time.perf_counter()
                    # skip saving checkpoint, no PSNR improvement
                    continue

                ckpt.psnr = psnr_value
                ckpt_mgr.save()

                self.now = time.perf_counter()

    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr = self.checkpoint.model(lr, training=True)
            loss_value = self.loss(hr, sr)

        gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables)
        self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables))

        return loss_value

    def evaluate(self, dataset):
        return evaluate(self.checkpoint.model, dataset)

    def restore(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f'Model restored from checkpoint at step {self.checkpoint.step.numpy()}.')


class EDSRTrainer(Trainer):
    def __init__(self,
                 model,
                 checkpoint_dir,
                 learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-4, 5e-5])):
        super().__init__(model, loss=MeanAbsoluteError(), learning_rate=learning_rate, checkpoint_dir=checkpoint_dir)

    def train(self, train_dataset, valid_dataset, steps=300000, evaluate_every=1000, save_best_only=True):
        super().train(train_dataset, valid_dataset, steps, evaluate_every, save_best_only)


class WDSRTrainer(Trainer):
    def __init__(self,
                 model,
                 checkpoint_dir,
                 learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-3, 5e-4])):
        super().__init__(model, loss=MeanAbsoluteError(), learning_rate=learning_rate, checkpoint_dir=checkpoint_dir)

    def train(self, train_dataset, valid_dataset, steps=300000, evaluate_every=1000, save_best_only=True):
        super().train(train_dataset, valid_dataset, steps, evaluate_every, save_best_only)


class SrganGeneratorTrainer(Trainer):
    def __init__(self,
                 model,
                 checkpoint_dir,
                 learning_rate=1e-4):
        super().__init__(model, loss=MeanSquaredError(), learning_rate=learning_rate, checkpoint_dir=checkpoint_dir)

    def train(self, train_dataset, valid_dataset, steps=1000000, evaluate_every=1000, save_best_only=True):
        super().train(train_dataset, valid_dataset, steps, evaluate_every, save_best_only)


class SRGANTrainer:
    # Optmizers for generator and discriminator. SRGAN will be trained for
    # 200,000 steps and learning rate is reduced from 1e-4 to 1e-5 after
    # 100,000 steps
    def __init__(self,
                 generator,
                 discriminator,
                 content_loss='VGG54',
                 learning_rate=PiecewiseConstantDecay(boundaries=[100000], values=[1e-4, 1e-5],),
                 checkpoint_dir='./ckpt/srgan'):

        if content_loss == 'VGG22':
            self.vgg = srgan.vgg_22()
        elif content_loss == 'VGG54':
            self.vgg = srgan.vgg_54()
        else:
            raise ValueError("content_loss must be either 'VGG22' or 'VGG54'")

        self.now = None
        self.content_loss = content_loss
        self.generator_optimizer = Adam(learning_rate=learning_rate)
        self.discriminator_optimizer = Adam(learning_rate=learning_rate)

        # Used in generator_loss and discriminator_loss
        self.binary_cross_entropy = BinaryCrossentropy(from_logits=False)
        # Used in content_loss
        self.mean_squared_error = MeanSquaredError()
        
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              psnr=tf.Variable(-1.0),
                                              generator_optimizer=Adam(learning_rate),
                                              discriminator_optimizer=Adam(learning_rate),
                                              generator=generator,
                                              discriminator=discriminator)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                             directory=checkpoint_dir,
                                                             max_to_keep=3)

        self.restore()


    @property
    def generator(self):
        """SRGAN generator."""
        return self.checkpoint.generator
    
    @property
    def discriminator(self):
        """SRGAN discriminator."""
        return self.checkpoint.discriminator
    
    def train(self, train_dataset, valid_dataset, steps=200000, evaluate_every=50):
        pls_metric = Mean()
        dls_metric = Mean()
        
        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint
        
        self.now = time.perf_counter()

        for lr, hr in train_dataset.take(steps - ckpt.step.numpy()):
            ckpt.step.assign_add(1)
            step = ckpt.step.numpy()

            pl, dl = self.train_step(lr, hr)
            pls_metric(pl)
            dls_metric(dl)

            if step % evaluate_every == 0:
                # Compute PSNR on validation dataset
                psnr_value, ssim_value = self.evaluate(valid_dataset)
                
                duration = time.perf_counter() - self.now
                print(f'{step}/{steps}, perceptual loss = {pls_metric.result():.4f}, discriminator loss = {dls_metric.result():.4f}, PSNR = {psnr_value.numpy():3f}, SSIM = {ssim_value.numpy():3f} ({duration:.2f}s)')
                
                pls_metric.reset_states()
                dls_metric.reset_states()

                ckpt.psnr = psnr_value
                ckpt_mgr.save()

                self.now = time.perf_counter()

     
    @tf.function
    def train_step(self, lr, hr):
        """SRGAN training step.
    
        Takes an LR and an HR image batch as input and returns
        the computed perceptual loss and discriminator loss.
        """
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            # Forward pass
            sr = self.checkpoint.generator(lr, training=True)
            hr_output = self.checkpoint.discriminator(hr, training=True)
            sr_output = self.checkpoint.discriminator(sr, training=True)

            # Compute losses
            con_loss = self._content_loss(hr, sr)
            gen_loss = self._generator_loss(sr_output)
            perc_loss = con_loss + 0.001 * gen_loss
            disc_loss = self._discriminator_loss(hr_output, sr_output)

        # Compute gradient of perceptual loss w.r.t. generator weights 
        gradients_of_generator = gen_tape.gradient(perc_loss, self.checkpoint.generator.trainable_variables)
        # Compute gradient of discriminator loss w.r.t. discriminator weights 
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.checkpoint.discriminator.trainable_variables)

        # Update weights of generator and discriminator
        self.checkpoint.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.checkpoint.generator.trainable_variables))
        self.checkpoint.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.checkpoint.discriminator.trainable_variables))

        return perc_loss, disc_loss
    
    def evaluate(self, dataset):
        return evaluate(self.checkpoint.generator, dataset)

    def restore(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f'Model restored from checkpoint at step {self.checkpoint.step.numpy()}.')
            
    @tf.function
    def _content_loss(self, hr, sr):
        sr = preprocess_input(sr)
        hr = preprocess_input(hr)
        sr_features = self.vgg(sr) / 12.75
        hr_features = self.vgg(hr) / 12.75
        return self.mean_squared_error(hr_features, sr_features)

    def _generator_loss(self, sr_out):
        return self.binary_cross_entropy(tf.ones_like(sr_out), sr_out)

    def _discriminator_loss(self, hr_out, sr_out):
        hr_loss = self.binary_cross_entropy(tf.ones_like(hr_out), hr_out)
        sr_loss = self.binary_cross_entropy(tf.zeros_like(sr_out), sr_out)
        return hr_loss + sr_loss
