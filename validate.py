import numpy as np
import matplotlib.pyplot as pl
from astropy.io import fits
import torch
import model
import pathlib
import time
from astropy.io import fits
import shutil
import os
import glob
from tqdm import tqdm
from skimage import measure
import scipy.ndimage as nd
from complex import complex_division, complex_multiply_astar_b
try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False

import scipy.optimize as opt

def twoD_GaussianScaledAmp(r, xo, yo, sigma_x, sigma_y, amplitude, offset):
    """Function to fit, returns 2D gaussian function as 1D array"""
    xo = float(xo)
    yo = float(yo)    
    x, y = r
    g = offset + amplitude*np.exp( - (((x-xo)**2)/(2*sigma_x**2) + ((y-yo)**2)/(2*sigma_y**2)))
    return g.ravel()

def getFWHM_GaussianFitScaledAmp(img):
    """Get FWHM(x,y) of a blob by 2D gaussian fitting
    Parameter:
        img - image as numpy array
    Returns: 
        FWHMs in pixels, along x and y axes.
    """
    x = np.linspace(0, img.shape[1], img.shape[1])
    y = np.linspace(0, img.shape[0], img.shape[0])
    x, y = np.meshgrid(x, y)
    #Parameters: xpos, ypos, sigmaX, sigmaY, amp, baseline
    initial_guess = (img.shape[1]/2,img.shape[0]/2,10,10,1,0)
    # subtract background and rescale image into [0,1], with floor clipping
    bg = np.percentile(img,5)
    img_scaled = np.clip((img - bg) / (img.max() - bg),0,1)
    popt, pcov = opt.curve_fit(twoD_GaussianScaledAmp, (x, y), 
                               img_scaled.ravel(), p0=initial_guess,
                               bounds = ((img.shape[1]*0.4, img.shape[0]*0.4, 1, 1, 0.5, -0.1),
                                     (img.shape[1]*0.6, img.shape[0]*0.6, img.shape[1]/2, img.shape[0]/2, 1.5, 0.5)))
    xcenter, ycenter, sigmaX, sigmaY, amp, offset = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]
    FWHM_x = np.abs(4*sigmaX*np.sqrt(-0.5*np.log(0.5)))
    FWHM_y = np.abs(4*sigmaY*np.sqrt(-0.5*np.log(0.5)))
    return (FWHM_x, FWHM_y)

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


class Deconvolution(object):
    
    def __init__(self, basis_wavefront='zernike', npix_image=128, n_modes=44, n_frames=10, gpu=0, corner=(0,0),\
        batch_size=16, checkpoint=None):

        self.pixel_size = 0.0303
        self.telescope_diameter = 256.0  # cm
        self.central_obscuration = 51.0  # cm
        self.wavelength = 8000.0
        self.n_frames = n_frames
        self.batch_size = batch_size
        
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

        if (checkpoint is None):
            files = glob.glob('trained/*.pth')
            self.checkpoint = max(files, key=os.path.getctime)
        else:
            self.checkpoint = '{0}'.format(checkpoint)

        print("=> loading checkpoint '{}'".format(self.checkpoint))

        tmp = torch.load(self.checkpoint, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(tmp['state_dict'])
        print("=> loaded checkpoint '{}'".format(self.checkpoint))

    def get_obs(self, obsfile, low, corner=None, align_frames=True):
        f = fits.open(obsfile)
        if (corner is None):
            im = f[0].data[low:low+self.n_frames, :, :].astype('float32')
        else:
            im = f[0].data[low:low+self.n_frames, corner[0]:corner[0]+128, corner[1]:corner[1]+128].astype('float32')

        tmp = np.sum(im, axis=0)

        delta = np.unravel_index(np.argmax(tmp), (128, 128))
        print(f"Delta_0 = {delta}")
        im = np.roll(im, (64-delta[0], 64-delta[1]), axis=(1, 2))

        if (align_frames):
            print("Aligning frames")
            for i in tqdm(range(self.n_frames-1)):
                sh = align(im[i+1, :, :], im[0, :, :])
                im[i+1, :, :] = nd.interpolation.shift(im[i+1,:,:], sh, mode='reflect')

        tmp = np.sum(im, axis=0)

        delta = np.unravel_index(np.argmax(tmp), (128, 128))
        print(f"Delta_1 = {delta}")
        im = np.roll(im, (64-delta[0], 64-delta[1]), axis=(1, 2))

        max_im = np.max(im)
        min_im = np.min(im)
        im = (im - min_im) / (max_im - min_im)


        ff = np.fft.fft2(im)
        im_fft = np.concatenate([ff.real[:, :, :, None], ff.imag[:, :, :, None]], axis=-1)

        return im, im_fft

    def validate(self, outfile, low, corner=None, align_frames=True):
        """
        Train for one epoch
        """

        # Set model in training mode
        self.model.eval()

        # Mask in Fourier plane
        x = np.linspace(-1, 1, self.npix_image)
        xx, yy = np.meshgrid(x, x)
        rho = np.sqrt(xx ** 2 + yy ** 2)
        mask = rho <= 0.5
        mask_simple = np.fft.fftshift(mask.astype('float32'))
        
        tmp = self.get_obs(outfile, low, corner, align_frames=align_frames)

        start = time.time()

        images = torch.tensor(tmp[0].astype('float32')[None, :, :])
        images_ft = torch.tensor(tmp[1].astype('float32')[None, :, :])
        variance = 1e-6*torch.ones(self.batch_size)
        images, images_ft, variance = images.to(self.device), images_ft.to(self.device), variance.to(self.device)

        with torch.no_grad():
            
            coeff, numerator, denominator, psf, psf_ft, loss = self.model(images, images_ft, variance)

        tmp = complex_multiply_astar_b(numerator, numerator)
        filt = 1.0 - complex_division(denominator, tmp)[..., 0]
        filt[filt < 0.2] = 0.0
        filt[filt > 1.0] = 1.0
        tmp = np.fft.fftshift(filt.cpu().numpy())
        all_contours = measure.find_contours(tmp[0,:,:], 0.01)
        origin = 128/2 * np.ones((1,2))
        index_contour = -1
        for i in range(len(all_contours)):
            is_inside = measure.points_in_poly(origin, all_contours[i])
            if (is_inside[0]):
                index_contour = i
                break

        if (index_contour != -1):
            mask = np.fft.fftshift(tmp[0, :, :] * measure.grid_points_in_poly((128, 128), all_contours[index_contour]))
        else:
            mask = np.copy(mask_simple)
        mask_torch = torch.tensor(mask).to(self.device)

        F = complex_division(numerator, denominator) * mask_torch[None, :, :, None]
        im = torch.ifft(F, 2)[:, :, :, 0]

        im = im.detach().cpu().numpy()

        if (np.isnan(im).sum() != 0):
            print("NaN detected in image. Defaulting to standard mask")
            mask_torch = torch.tensor(mask_simple).to(self.device)

            F = complex_division(numerator, denominator) * mask_torch[None, :, :, None]
            im = torch.ifft(F, 2)[:, :, :, 0]

            im = im.detach().cpu().numpy()

        coeff = coeff.detach().cpu().numpy()
        im[im < 0] = 0.0
        psf = psf[..., 0].detach().cpu().numpy()
        images = images[0, :, :, :].detach().cpu().numpy()

        final = time.time()

        print(f"Elapsed time : {final-start} s")

        im_ft = np.fft.fft2(im[0,:,:])
        psf_ft = np.fft.fft2(psf, axes=(1,2))
        im_degraded = np.fft.ifft2(im_ft[None, :, :] * psf_ft).real

        psf = np.fft.fftshift(psf[:,:,:])

        return images, im_degraded, psf, im, coeff, loss

if (__name__ == '__main__'):

    #pl.close('all')
    gpu = 3
    align_frames = True
    saveplot = False

    for figure in [3]:
        pl.close('all')
        if (figure == 1):
            outfile = '/scratch1/aasensio/fastcam/verification/RAW_GJ661_1000_030g1m20_I0.fits'
            low = 0
            corner = (0, 0)
            name = 'GJ661'
            label = 'GJ661'

        if (figure == 2):
            outfile = '/scratch1/aasensio/fastcam/verification/cubo_sel_alig.fits'
            low = 0
            corner = (146, 225)
            name = 'GJ569'
            label = 'GJ569'

        if (figure == 3):
            outfile = '/scratch1/aasensio/fastcam/verification/RAW_SIG_ORI_AB_1000_030g1m1.fits'
            low = 1
            corner = (0, 0)
            name = 'SIGORI'
            label = r'$\sigma$-ori'

        if (figure == 4):
            outfile = '/scratch1/aasensio/fastcam/verification/RAW_GJ856_1000_030g1m2_I0.fits'
            low = 1
            corner = (0, 0)
            name = 'GJ856'
            label = 'GJ856'

        if (figure == 5):
            outfile = '/scratch1/aasensio/fastcam/verification/RAW_M15_33_1000_030g1m20.fits'
            low = 1
            corner = (0, 0)
            name = 'M15'
            label = 'M15'
        
        basis = 'kl'
        
        checkpoint = 'trained/2020-04-25-13:03.pth'
        checkpoint = 'trained/2020-04-30-17:11.pth'
        checkpoint = 'trained/2020-05-05-15:54.pth'  # best

        checkpoint = 'trained/2020-05-18-14:13.pth' # decent, trained with single star
        checkpoint = 'trained/2020-05-18-18:28.pth' # trained with 100 steps. Looks promising

        checkpoint = 'trained/2020-05-21-11:44.pth' # trained with 100 steps. Looks promising

        out = Deconvolution(basis_wavefront=basis, n_modes=55, gpu=gpu, n_frames=100, batch_size=1, checkpoint=checkpoint)
        images, im_degraded, psf, im, coeff, loss = out.validate(outfile, low, corner, align_frames=align_frames)
        sn = np.max(im[0,:,:]) / np.std(im[0,100:,100:])
        f, ax = pl.subplots(nrows=3, ncols=6, figsize=(5*2,5), sharex=True, sharey=True)#, \
        # gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
            
        for i in range(6):
            ax[0,i].imshow(images[i, :, :])
            ax[1,i].imshow(psf[i, :, :])
            ax[2,i].imshow(im_degraded[i, :, :])
            for j in range(3):
                ax[j,i].set_xticklabels([])
                ax[j,i].set_yticklabels([])
                ax[j,i].set_xticks([])
                ax[j,i].set_yticks([])

        ax[0,0].set_title(f'{label}', fontsize=9, weight='bold')
        ax[0,0].text(5, 120, 'Original', color='white', fontsize=9, weight='bold')
        ax[1,0].text(5, 120, 'PSF', color='white', fontsize=9, weight='bold')
        ax[2,0].text(5, 120, 'Deconv+degraded', color='white', fontsize=9, weight='bold')
        pl.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.03, wspace=0.05, hspace=0.05)
        #pl.tight_layout()
        
        pl.show()
        
        if (saveplot):
            pl.savefig(f'{name}_psfs.pdf')

        f, ax = pl.subplots(nrows=4, ncols=4, sharex=True, figsize=(10,10))
        for i in range(16):
            ax.flat[i].plot(coeff[:, i])
            ax.flat[i].set_title(f'KL({i+1})')
            ax.flat[i].set_ylim([-1.5,1.5])
        for i in range(4):
            ax[-1,i].set_xlabel('Frame')
            ax[i,0].set_ylabel(r'$\alpha$ [rad]')
        pl.tight_layout()
        pl.show()
        if (saveplot):
            pl.savefig(f'{name}_zernike.pdf')
            
        out = Deconvolution(basis_wavefront=basis, n_modes=55, gpu=gpu, n_frames=20, batch_size=1, checkpoint=checkpoint)
        images, im_degraded, psf, im20, _, loss20 = out.validate(outfile, low, corner, align_frames=align_frames)
        sn20 = np.max(im20[0,:,:]) / np.std(im20[0,100:,100:])

        out = Deconvolution(basis_wavefront=basis, n_modes=55, gpu=gpu, n_frames=50, batch_size=1, checkpoint=checkpoint)
        images, im_degraded, psf, im50, _, loss50 = out.validate(outfile, low, corner, align_frames=align_frames)
        sn50 = np.max(im50[0,:,:]) / np.std(im50[0,100:,100:])

        out = Deconvolution(basis_wavefront=basis, n_modes=55, gpu=gpu, n_frames=200, batch_size=1, checkpoint=checkpoint)
        images, im_degraded, psf, im200, _, loss200 = out.validate(outfile, low, corner, align_frames=align_frames)
        sn200 = np.max(im200[0,:,:]) / np.std(im200[0,100:,100:])

        # out = Deconvolution(basis_wavefront=basis, n_modes=55, n_frames=100, batch_size=1, checkpoint=checkpoint)
        # images, im_degraded, psf, im500, _, loss500 = out.validate(outfile, low, corner)

        f, ax = pl.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(5.5,5.5))
        ax[0,0].imshow(im20[0,:,:])
        ax[0,0].text(5, 120, '20 frames', color='white', fontsize=9, weight='bold')
        ax[0,1].imshow(im50[0,:,:])
        ax[0,1].text(5, 120, '50 frames', color='white', fontsize=9, weight='bold')
        ax[1,0].imshow(im[0,:,:])
        ax[1,0].text(5, 120, '100 frames', color='white', fontsize=9, weight='bold')
        ax[1,1].imshow(im200[0,:,:])
        ax[1,1].text(5, 120, '200 frames', color='white', fontsize=9, weight='bold')
        for i in range(2):
            for j in range(2):
                ax[j,i].set_xticklabels([])
                ax[j,i].set_yticklabels([])
                ax[j,i].set_xticks([])
                ax[j,i].set_yticks([])
        ax[0,0].set_title(f'{label}', fontsize=9, weight='bold')
        pl.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.95, wspace=0.05, hspace=0.05)
        pl.show()
        if (saveplot):
            pl.savefig(f'{name}_deconvolved.pdf')

        # f, ax = pl.subplots()
        # ax.imshow(im[0,:,:])
        # pl.tight_layout()
        # pl.show()

        #print(f'loss(5) = {loss5}')
        #print(f'loss(10) = {loss10}')
        #print(f'loss(20) = {loss20}')
        #print(f'loss(100) = {loss100}')
        print(f'---------- {name}------------')
        print(f'loss(100) = {loss}')
        FWHM_x, FWHM_y = getFWHM_GaussianFitScaledAmp(im[0,:,:])
        FWHM = np.sqrt(0.5*(FWHM_x**2 + FWHM_y**2))
        FWHM_diff = 1.22 * 800e-9 / 2.56 * 206265 
        print(f'FWHM_x={out.pixel_size * FWHM_x}')
        print(f'FWHM_y={out.pixel_size * FWHM_y}')
        print(f'FWHM  ={out.pixel_size * FWHM}={out.pixel_size * FWHM / FWHM_diff}')
        print(f'FWHM_diff = {FWHM_diff}')

        snframe = np.max(images[0,:,:]) / np.std(images[0,100:,100:])
        print(f'S/N_frame={snframe}')
        print(f'S/N_20={sn20}')
        print(f'S/N_50={sn50}')
        print(f'S/N_100={sn}')
        print(f'S/N_200={sn200}')

        ff = open(f'{name}_info.txt', 'w')
        ff.write(f'FWHM  {out.pixel_size * FWHM}\n')
        ff.write(f'FWHM_diff_units {out.pixel_size * FWHM / FWHM_diff}\n')
        ff.write(f'S/N_frame {snframe}\n')
        ff.write(f'S/N_20 {sn20}\n')
        ff.write(f'S/N_50 {sn50}\n')
        ff.write(f'S/N_100 {sn}\n')
        ff.write(f'S/N_200 {sn200}\n')
        

        # Photometry
        if (figure == 4 or figure == 5):
            if (figure == 4):
                
                flux2 = np.sum(images[:,60:,0:40],axis=(1,2)) 
                flux1 = np.sum(images[:,40:88,40:88],axis=(1,2))

                flux2_dec_20 = np.sum(im20[0,60:,0:40])
                flux1_dec_20 = np.sum(im20[0,40:88,40:88])

                flux2_dec_50 = np.sum(im50[0,60:,0:40])
                flux1_dec_50 = np.sum(im50[0,40:88,40:88])  

                flux2_dec_100 = np.sum(im[0,60:,0:40])
                flux1_dec_100 = np.sum(im[0,40:88,40:88])

                flux2_dec_200 = np.sum(im200[0,60:,0:40])
                flux1_dec_200 = np.sum(im200[0,40:88,40:88]) 

            if (figure == 5):
                
                flux2 = np.sum(images[:,40:90,40:90],axis=(1,2)) 
                flux1 = np.sum(images[:,0:40,0:40],axis=(1,2))

                flux2_dec_20 = np.sum(im20[0,40:90,40:90])
                flux1_dec_20 = np.sum(im20[0,0:40,0:40])

                flux2_dec_50 = np.sum(im50[0,40:90,40:90])
                flux1_dec_50 = np.sum(im50[0,0:40,0:40])  

                flux2_dec_100 = np.sum(im[0,40:90,40:90])
                flux1_dec_100 = np.sum(im[0,0:40,0:40])

                flux2_dec_200 = np.sum(im200[0,40:90,40:90])
                flux1_dec_200 = np.sum(im200[0,0:40,0:40]) 

            f, ax = pl.subplots()
            ax.plot(flux1 / flux2)
            ax.axhline(np.mean(flux1 / flux2), color='C0', label='Average')
            ax.axhline(flux1_dec_20 / flux2_dec_20, color='C1', label='20 images')
            ax.axhline(flux1_dec_50 / flux2_dec_50, color='C2', label='50 images')
            ax.axhline(flux1_dec_100 / flux2_dec_100, color='C3', label='100 images')
            ax.axhline(flux1_dec_200 / flux2_dec_200, color='C4', label='200 images')
            ax.set(xlabel='Frame', ylabel='Flux ratio', title=name)
            pl.legend()
            pl.show()
            pl.savefig(f'{name}_flux.pdf')

        if (os.path.isfile(f'../classic/{name}.npz')):
            tmp = np.load(f'../classic/{name}.npz')
            psf = tmp['psf']
            image = tmp['image']
            coeff = tmp['coeff']

            im_ft = np.fft.fft2(image)
            psf_ft = np.fft.fft2(psf, axes=(1,2))
            im_degraded = np.fft.ifft2(im_ft[None, :, :] * psf_ft).real

            psf = np.fft.fftshift(psf[:,:,:])

            f, ax = pl.subplots(nrows=3, ncols=6, figsize=(5*2,5), sharex=True, sharey=True)#, \
            # gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
                
            for i in range(6):
                ax[0,i].imshow(images[i, :, :])
                ax[1,i].imshow(psf[i, :, :])
                ax[2,i].imshow(im_degraded[i, :, :])
                for j in range(3):
                    ax[j,i].set_xticklabels([])
                    ax[j,i].set_yticklabels([])
                    ax[j,i].set_xticks([])
                    ax[j,i].set_yticks([])

            ax[0,0].set_title(f'{label}', fontsize=9, weight='bold')
            ax[0,0].text(5, 120, 'Original', color='white', fontsize=9, weight='bold')
            ax[1,0].text(5, 120, 'PSF - Classic', color='white', fontsize=9, weight='bold')
            ax[2,0].text(5, 120, 'Deconv+degraded', color='white', fontsize=9, weight='bold')
            pl.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.03, wspace=0.05, hspace=0.05)
            #pl.tight_layout()
            
            pl.show()
            pl.savefig(f'{name}_classic_psfs.pdf')

            f, ax = pl.subplots(nrows=4, ncols=4, sharex=True, figsize=(10,10))
            for i in range(16):
                ax.flat[i].plot(coeff[:, i])
                ax.flat[i].set_title(f'KL({i+1})')
                ax.flat[i].set_ylim([-1.5,1.5])
            for i in range(4):
                ax[-1,i].set_xlabel('Frame')
                ax[i,0].set_ylabel(r'$\alpha$ [rad]')
            pl.tight_layout()
            pl.show()

            pl.savefig(f'{name}_classic_zernike.pdf')

            f, ax = pl.subplots()
            ax.imshow(image)
            ax.set_title(f'{label}', fontsize=9, weight='bold')

            pl.show()
            pl.savefig(f'{name}_classic_deconvolved.pdf')

            FWHM_x, FWHM_y = getFWHM_GaussianFitScaledAmp(image)
            FWHM = np.sqrt(0.5*(FWHM_x**2 + FWHM_y**2))
            FWHM_diff = 1.22 * 800e-9 / 2.56 * 206265 
            print(f'FWHM_x={out.pixel_size * FWHM_x}')
            print(f'FWHM_y={out.pixel_size * FWHM_y}')
            print(f'FWHM  ={out.pixel_size * FWHM}={out.pixel_size * FWHM / FWHM_diff}')

            ff.write(f'FWHM  {out.pixel_size * FWHM}\n')
            ff.write(f'FWHM_diff_units {out.pixel_size * FWHM / FWHM_diff}\n')
        
        ff.close()
