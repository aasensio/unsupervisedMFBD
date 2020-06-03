import numpy as np
import matplotlib.pyplot as pl
import h5py
import torch
import model
import pathlib
import time
import shutil
import os
from tqdm import tqdm
import argparse
import scipy.ndimage as nd
try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Save neural network state
    
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename+'.best')

def align(a, b):        

    if(a.shape[0] != b.shape[0] or a.shape[1] != b.shape[1]):
        print("align: ERROR, both images must have the same size")
        return(0.0,0.0)
    
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)

    cc = np.roll(np.roll(np.real(np.fft.ifft2(fa.conjugate() * fb)), -int(fa.shape[0]//2), axis=0), -int(fa.shape[1]//2), axis=1)
    
    mm = np.argmax(cc)
    xy = ( mm // fa.shape[1], mm % fa.shape[1])

    cc = cc[xy[0]-1:xy[0]+2, xy[1]-1:xy[1]+2]

    y = 2.0*cc[1,1]
    x = (cc[1,0]-cc[1,2])/(cc[1,2]+cc[1,0]-y)*0.5
    y = (cc[0,1]-cc[2,1])/(cc[2,1]+cc[0,1]-y)*0.5

    x += xy[1] - fa.shape[1]//2
    y += xy[0] - fa.shape[0]//2

    return(y,x)

class Dataset(torch.utils.data.Dataset):
    """
    Dataset

      Scripts to produce the training sets : db.py
    
    """
    def __init__(self, filename, n_training_per_star=200, n_frames=10, validation=False):
        super(Dataset, self).__init__()
        
        # Read the video with the images
        self.filename = filename
        self.f = h5py.File(self.filename, 'r')
        self.datasets = [i for i in self.f.keys()]
        # if (not validation):
            # ind = [1, 13]
            # self.datasets = [self.datasets[i] for i in ind]
        self.n_training_per_star = n_training_per_star
        self.n_datasets = len(self.datasets)
        self.n_training = self.n_datasets * self.n_training_per_star
        self.n_frames = n_frames

        self.ind_time = []
        self.ind_dataset = []

        x, y = np.arange(128), np.arange(128)
        self.xx, self.yy = np.meshgrid(x, y)

        for dset in self.datasets:
            n, _ = self.f[dset].shape
            ind_time = np.random.randint(low=0, high=n-self.n_frames, size=self.n_training_per_star)
            self.ind_dataset.extend([dset] * self.n_training_per_star)
            self.ind_time.extend(ind_time)
        
        print(f"Number of training examples of {self.filename}: {self.n_training}")
                
    def __getitem__(self, index):
        dset = self.ind_dataset[index]
        low = self.ind_time[index]
        high = self.ind_time[index] + self.n_frames
        im = self.f[dset][low:high, :].reshape((self.n_frames, 128, 128))


        rot = np.random.randint(low=0, high=4, size=1)
        flipx = np.random.randint(low=0, high=2, size=1)
        flipy = np.random.randint(low=0, high=2, size=1)
        
        im = np.rot90(im, rot[0], axes=(1,2))
        if (flipx[0] == 1):
            im = im[:, ::-1, :]
        if (flipy[0] == 1):
            im = im[:, :, ::-1]

        max_im = np.max(im)
        min_im = np.min(im)
        
        im = (im - min_im) / (max_im - min_im)

        # im_aligned = np.zeros_like(im)
        # im_aligned[0, :, :] = im[0, :, :]
        # for i in range(self.n_frames-1):
        #     sh = align(im[i, :, :], im[i+1, :, :])
        #     im_aligned[i+1, :, :] = nd.interpolation.shift(im[i+1,:,:], sh, mode='wrap')

        # im = np.copy(im_aligned)

        # Make sure that the average is again at the center of the FOV
        tmp = np.sum(im, axis=0)

        delta = np.unravel_index(np.argmax(tmp), (128, 128))
        im = np.roll(im, (64-delta[0], 64-delta[1]), axis=(1, 2))

        ff = np.fft.fft2(im)
        im_fft = np.concatenate([ff.real[:, :, :, None], ff.imag[:, :, :, None]], axis=-1)

        variance = np.var(im[:, 0:10, 0:10])
        
        return im, im_fft, variance
        
    def __len__(self):
        return self.n_training

class Deconvolution(object):
    
    def __init__(self, basis_wavefront='zernike', npix_image=128, n_modes=44, n_frames=10, gpu=0, smooth=0.05,\
        batch_size=16, arguments=None):

        self.pixel_size = 0.0303
        self.telescope_diameter = 256.0  # cm
        self.central_obscuration = 51.0  # cm
        self.wavelength = 8000.0
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.arguments = arguments
        
        self.basis_for_wavefront = basis_wavefront
        self.npix_image = npix_image
        self.n_modes = n_modes
        self.gpu = gpu
        self.cuda = torch.cuda.is_available()
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")

        # Ger handlers to later check memory and usage of GPUs
        if (NVIDIA_SMI):
            nvidia_smi.nvmlInit()
            self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.gpu)
            print("Computing in {0} : {1}".format(gpu, nvidia_smi.nvmlDeviceGetName(self.handle)))

        # Define the neural network model
        print("Defining the model...")
        self.model = model.Network(device=self.device, n_modes=self.n_modes, n_frames=self.n_frames, \
            pixel_size=self.pixel_size, telescope_diameter=self.telescope_diameter, central_obscuration=self.central_obscuration, wavelength=self.wavelength,\
            basis_for_wavefront=self.basis_for_wavefront, npix_image=self.npix_image).to(self.device)

        print('N. total parameters : {0}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

        kwargs = {'num_workers': 1, 'pin_memory': False} if self.cuda else {}
        # Data loaders that will inject data during training
        self.training_dataset = Dataset(filename='/scratch1/aasensio/fastcam/training_small.h5', n_training_per_star=1000, n_frames=self.n_frames)
        self.train_loader = torch.utils.data.DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, **kwargs)

        self.validation_dataset = Dataset(filename='/scratch1/aasensio/fastcam/validation_small.h5', n_training_per_star=100, n_frames=self.n_frames, validation=True)
        self.validation_loader = torch.utils.data.DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, **kwargs)

    def init_optimize(self, lr=3e-4, smooth=0.05):
        """
        Initialize the training
        """

        self.lr = lr
        self.smooth = smooth

        print('Learning rate : {0}'.format(self.lr))

        # Create directory with trained outputs if it does not exist
        p = pathlib.Path('trained/')
        p.mkdir(parents=True, exist_ok=True)
        
        # Get output file (it uses the time for getting a unique file)
        current_time = time.strftime("%Y-%m-%d-%H:%M")
        self.out_name = 'trained/{0}'.format(current_time)

        # Copy model to keep track of the specificities of the trained model
        shutil.copyfile(model.__file__, '{0}.model.py'.format(self.out_name))
        shutil.copyfile('{0}/{1}'.format(os.path.dirname(os.path.abspath(__file__)), __file__), '{0}_trainer.py'.format(self.out_name))
                
        # Save learning rate and weight decay if used
        f = open('{0}_args.dat'.format(self.out_name), 'w')
        f.write(str(self.arguments))
        f.close()
        
        # Instantiate optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)

        # Instantiate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.5)
        
    def optimize(self, n_epochs):
        """
        Do the optimization
        """

        self.n_epochs = n_epochs
        self.loss = []
        self.loss_val = []
        best_loss = -1e10        
        
        trainF = open('{0}.loss.csv'.format(self.out_name), 'w')

        print('Model : {0}'.format(self.out_name))

        epoch_modes = -1

        # Loop over epochs
        for epoch in range(1, self.n_epochs + 1):

            # Do one epoch for the training set
            self.train(epoch)

            # Do one epoch for the validation set
            self.validate(epoch)

            # Update learning rate if needed
            self.scheduler.step()

            # Save information about the state of the neural network
            trainF.write('{},{},{}\n'.format(
                epoch, self.loss[-1], self.loss_val[-1]))
            trainF.flush()

            is_best = self.loss_val[-1] < best_loss
            best_loss = max(self.loss_val[-1], best_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_loss': best_loss,
                'optimizer': self.optimizer.state_dict(),
            }, is_best, filename='{0}.pth'.format(self.out_name))
            
        trainF.close()
        

    def train(self, epoch):
        """
        Train for one epoch
        """

        # Set model in training mode
        self.model.train()

        print("Epoch {0}/{1}".format(epoch, self.n_epochs))
        t = tqdm(self.train_loader)
        loss_avg = 0.0
        
        # Get current learning rate
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']

        for batch_idx, (images, images_ft, variance) in enumerate(t):
            
            # Move all data to GPU/CPU
            images, images_ft, variance = images.to(self.device), images_ft.to(self.device), variance.to(self.device)

            # Zero the gradients in the optimizer
            self.optimizer.zero_grad()

            
            # Evaluate the model
            coeff, numerator, denominator, psf, psf_ft, loss = self.model(images, images_ft, variance)            
                    
            # Backpropagate
            loss.backward()

            if (batch_idx == 0):
                loss_avg = loss.item()
            else:
                loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg

            # Update the weights according to the optimizer
            self.optimizer.step()

            # Get GPU usage for printing
            gpu_usage = ''
            memory_usage = ''
            if (NVIDIA_SMI):
                tmp = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle)
                gpu_usage = gpu_usage+f' {tmp.gpu}'
                memory_usage = memory_usage+f' {tmp.memory}'

                t.set_postfix(loss=loss.item(), loss_avg=loss_avg, lr=current_lr, gpu=gpu_usage, mem=memory_usage)
            else:
                t.set_postfix(loss=loss.ite(), loss_avg=loss_avg, lr=current_lr)
                            
        self.loss.append(loss_avg)

    def validate(self, epoch):
        """
        Train for one epoch
        """

        # Set model in training mode
        self.model.eval()

        t = tqdm(self.validation_loader)
        loss_avg = 0.0
    
        with torch.no_grad():
            
            for batch_idx, (images, images_ft, variance) in enumerate(t):
                
                # Move all data to GPU/CPU
                images, images_ft, variance = images.to(self.device), images_ft.to(self.device), variance.to(self.device)

                # Evaluate the model
                coeff, numerator, denominator, psf, psf_ft, loss = self.model(images, images_ft, variance)

                if (batch_idx == 0):
                    loss_avg = loss.item()
                else:
                    loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg

                t.set_postfix(loss=loss_avg)
                            
        self.loss_val.append(loss_avg)

if (__name__ == '__main__'):

    parser = argparse.ArgumentParser(description='Train VAE')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                    metavar='LR', help='Learning rate')
    parser.add_argument('--gpu', '--gpu', default=1, type=int,
                    metavar='GPU', help='GPU')
    parser.add_argument('--smooth', '--smoothing-factor', default=0.05, type=float,
                    metavar='SM', help='Smoothing factor for loss')
    parser.add_argument('--epochs', '--epochs', default=50, type=int,
                    metavar='EPOCHS', help='Number of epochs')
    parser.add_argument('--frames', '--frames', default=5, type=int,
                    metavar='FRAMES', help='Number of frames')
    parser.add_argument('--modes', '--modes', default=120, type=int,
                    metavar='MODES', help='Number of modes')
    parser.add_argument('--batch', '--batch', default=16, type=int,
                    metavar='BATCH', help='Batch size')
    
    parsed = vars(parser.parse_args())

    print(parsed)

    out = Deconvolution(basis_wavefront='kl', n_modes=parsed['modes'], n_frames=parsed['frames'], smooth=parsed['smooth'], batch_size=parsed['batch'], gpu=parsed['gpu'], arguments=parsed)
    out.init_optimize(lr=parsed['lr'])
    out.optimize(n_epochs=parsed['epochs'])
    