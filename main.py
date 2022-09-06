import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from data import DIV2K
from model.edsr import edsr, edsr_weightnorm
from model.srgan import discriminator
from train import EDSRTrainer, SRGANTrainer
from model import resolve_single
from utils import load_image, plot_sample


# set_memory_growth() allocates exclusively the GPU memory needed
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
if len(physical_devices) is not 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Number of residual blocks
depth = 16

# Super-resolution factor
scale = 2

# Downgrade operator
downgrade = 'unknown'

# 
model = 'srgan'

# Location of model weights (needed for demo)
weights_dir = f'weights/{model}-{depth}-x{scale}-{downgrade}'
weights_file = os.path.join(weights_dir, 'weights.h5')

os.makedirs(weights_dir, exist_ok=True)

div2k_train = DIV2K(scale=scale, subset='train', downgrade=downgrade)
div2k_valid = DIV2K(scale=scale, subset='valid', downgrade=downgrade)
div2k_test = DIV2K(scale=scale, subset='test', downgrade=downgrade)

train_ds = div2k_train.dataset(batch_size=16, random_transform=True)
valid_ds = div2k_valid.dataset(batch_size=1, random_transform=False, repeat_count=1)
test_ds = div2k_test.dataset(batch_size=1, random_transform=False, repeat_count=1)

# ==================================================================================================

# # EDSR model
# trainer = EDSRTrainer(model=edsr(scale=scale, num_res_blocks=depth), 
#                       checkpoint_dir=f'.ckpt/edsr-{depth}-x{scale}-{downgrade}')

# # Train EDSR model for 10,000 steps (due to resource constraints) and evaluate model
# # every 250 steps on  the DIV2K validation set (10 images). Save a checkpoint only 
# # if evaluation PSNR has improved.
# trainer.train(train_ds,
#               valid_ds,
#               steps=20000, 
#               evaluate_every=250, 
#               save_best_only=True)

# # Restore from checkpoint with highest PSNR
# trainer.restore()

# # Evaluate model on test set (90 images)
# psnrv, ssimv = trainer.evaluate(test_ds)
# print(f'PSNR = {psnrv.numpy():3f}, SSIM = {ssimv.numpy():3f}')

# # Save weights to separate location (needed for demo)
# trainer.model.save_weights(weights_file)

# ==================================================================================================

# # # EDSR model
trainer = EDSRTrainer(model=edsr_weightnorm(scale=scale, num_res_blocks=depth), 
                      checkpoint_dir=f'.ckpt/{model}-{depth}-x{scale}-{downgrade}',
                      learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-3, 5e-4]))

# Train EDSR model for 10,000 steps (due to resource constraints) and evaluate model
# every 250 steps on  the DIV2K validation set (10 images). Save a checkpoint only 
# if evaluation PSNR has improved.
trainer.train(train_ds,
              valid_ds,
              steps=10000, 
              evaluate_every=250, 
              save_best_only=True)

# Restore from checkpoint with highest PSNR
trainer.restore()

# Evaluate model on test set (90 images)
psnrv, ssimv = trainer.evaluate(test_ds)
print(f'PSNR = {psnrv.numpy():3f}, SSIM = {ssimv.numpy():3f}')

# Save weights to separate location (needed for demo)
trainer.model.save_weights(weights_file)

# ==================================================================================================

# # EDSR model used as generator in SRGAN
# gan_generator = edsr(scale=scale, num_res_blocks=depth)
# gan_generator.load_weights(weights_file)

# gan_trainer = SRGANTrainer(generator=gan_generator, 
#                            discriminator=discriminator(), 
#                            checkpoint_dir=f'.ckpt/srgan-{depth}-x{scale}-{downgrade}')

# gan_trainer.train(train_ds, 
#                   valid_ds,
#                   steps=10000,
#                   evaluate_every=250)

# # Restore from checkpoint with highest PSNR
# gan_trainer.restore()

# # Evaluate model on test set (90 images)
# psnrv, ssimv = gan_trainer.evaluate(test_ds)
# print(f'PSNR = {psnrv.numpy():3f}, SSIM = {ssimv.numpy():3f}')


# # Save weights to separate location (needed for demo)
# gan_trainer.generator.save_weights(os.path.join(weights_dir, 'weights-fine-tuned.h5'))
# gan_trainer.discriminator.save_weights(os.path.join(weights_dir, 'weights-discriminator.h5'))

# ==================================================================================================

# Demo
# model = edsr(scale=scale, num_res_blocks=depth)
# model.load_weights(weights_file)

# def resolve_and_plot(lr_image_path):
#     lr = load_image(lr_image_path)
#     sr = resolve_single(model, lr)
#     plot_sample(lr, sr)
    
# resolve_and_plot('./demo/0851x4-crop.png') 