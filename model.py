import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.init as init
import util
import zern
import kl_modes
from complex import complex_multiply_astar_b, complex_division

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

class ConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size=3, stride=1, bn=True, activation=True):
        """Convolutional block : BN+RELU+CONV
        The CONV uses reflection padding
        BN and RELU can be on/off depending on the keywords "bn" and "activation"
        
        Args:
            inplanes (int): number of input channels
            outplanes (int): number of output channels
            kernel_size (int, optional): Kernel size. Defaults to 3.
            stride (int, optional): Stride. Defaults to 1.
            bn (bool, optional): Use batch normalization. Defaults to True.
            activation (bool, optional): Use activation. Defaults to True.
        """
        super(ConvBlock, self).__init__()

        self.use_bn = bn
        self.use_activation = activation

        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride)
        self.reflection = nn.ReflectionPad2d(int((kernel_size-1)/2))

        if (bn):
            self.bn = nn.BatchNorm2d(inplanes)

        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        if (self.use_bn):
            out = self.bn(x)
            out = self.elu(out)
            out = self.reflection(out)
            out = self.conv(out)

        else:
            out = self.reflection(x)
            out = self.conv(out)
            if (self.use_activation):
                out = self.relu(out)

        return out

class Recurrentnet(nn.Module):
    def __init__(self, in_planes, device, n_frames, npix_image, n, n_modes, n_lstm):
        """Neural network to estimate the wavefront coefficients from a set of images
        This one uses a recurrent architecture and works for M pairs of focused+defocused
        images. An encoder is applied in parallel to all pairs of images so that a vector
        of 128 features is obtained from each pair of images. These vectors are then fed
        into an bi-directional LSTM that provides as output another vector of size 128
        per timestep. A final linear layer projects these vectors into the modal coefficients.
        
        Args:
            in_planes (int): number of input images
            n_frames (int): number of frames
            npix_image (int): number of pixels of the image
            n (int, optional): Number of channels in the hidden convolutional layers. Defaults to 32.
            n_modes (int, optional): Number of output modes. Defaults to 40.
        """
        super(Recurrentnet, self).__init__()

        self.n_modes = n_modes
        self.npix_image = npix_image
        self.n_frames = n_frames
        self.n_lstm = n_lstm
        self.device = device

        self.A01 = ConvBlock(1, n, kernel_size=9, bn=False, activation=False)

        self.C01 = ConvBlock(n, n, kernel_size=7, stride=2)
        self.C02 = ConvBlock(n, n, kernel_size=7)
        self.C03 = ConvBlock(n, n, kernel_size=7)
        self.C04 = ConvBlock(n, n, kernel_size=7)

        self.C11 = ConvBlock(n, n, kernel_size=5, stride=2)
        self.C12 = ConvBlock(n, n, kernel_size=5)
        self.C13 = ConvBlock(n, n, kernel_size=5)
        self.C14 = ConvBlock(n, n, kernel_size=5)

        self.C21 = ConvBlock(n, n, kernel_size=3, stride=2)
        self.C22 = ConvBlock(n, n, kernel_size=3)
        self.C23 = ConvBlock(n, n, kernel_size=3)
        self.C24 = ConvBlock(n, n, kernel_size=3)

        kernel_size = 16

        self.C41 = nn.Conv2d(n, self.n_lstm, kernel_size=kernel_size, stride=1)
        
        self.C42 = nn.Linear(2*self.n_lstm, self.n_lstm)
        self.C43 = nn.Linear(self.n_lstm, n_modes)
        
        self.elu = nn.ELU()

        self.lstm = nn.LSTM(self.n_lstm, self.n_lstm, batch_first=True, bidirectional=True, dropout=0.0)

        # x = np.linspace(0, 1, self.npix_image)
        # y = np.linspace(0, 1, self.npix_image)
        # xx, yy = np.meshgrid(x, y)
        # self.xx = torch.tensor(xx.astype('float32')).to(self.device)
        # self.yy = torch.tensor(yy.astype('float32')).to(self.device)

    def weights_init(self):
        for module in self.modules():
            kaiming_init(module)

    def forward(self, images):
        
        # We reform the tensor from (B,Nf,1,nx,ny) to (B*Nf,1,nx,ny) so that the features
        # are extracted for all frames of all batches in parallel
        # B is the batch size
        # Nf is the number of frames
        tmp = images.view(-1, 1, self.npix_image, self.npix_image)

        # n_batch = tmp.size(0)

        # xx = self.xx[None, None, :, :].expand(n_batch, 1, self.npix_image, self.npix_image)
        # yy = self.yy[None, None, :, :].expand(n_batch, 1, self.npix_image, self.npix_image)

        # tmp = torch.cat([tmp, xx, yy], dim=1)

        # (B*Nf,2,129,128) -> (B*Nf,32,128,128)
        A01 = self.A01(tmp)

        # (B*Nf,32,128,128) -> (B*Nf,32,64,64)
        C01 = self.C01(A01)
        C02 = self.C02(C01)
        C03 = self.C03(C02)
        C04 = C01 + self.C04(C03)

        # (B*Nf,32,64,64) -> (B*Nf,32,32,32)
        C11 = self.C11(C04)
        C12 = self.C12(C11)
        C13 = self.C13(C12)
        C14 = C11 + self.C14(C13)

        # (B*Nf,32,32,32) -> (B*Nf,32,16,16)
        C21 = self.C21(C14)
        C22 = self.C22(C21)
        C23 = self.C23(C22)
        C24 = C21 + self.C24(C23)

        # (B*Nf,32,16,16) -> (B*Nf,128,1,1)
        out = self.C41(C24)

        # (B*Nf,128,1) -> (B*Nf,128)
        out = out.squeeze()

        # (B*Nf,128) -> (B,Nf,128)
        out = out.view(-1, self.n_frames, self.n_lstm)

        # (B,Nf,128) -> (B,Nf,128)
        out, _ = self.lstm(out)

        # (B,Nf,128) -> (B*Nf,128)
        out = out.reshape(-1, 2*self.n_lstm)

        # (B*Nf,128) -> (B*Nf,44)
        out = self.elu(self.C42(out))
        out = self.C43(out)

        # (B*Nf,N_modes) -> (B,Nf*N_modes)
        # out = out.view(-1, self.n_frames * self.n_modes)

        return out

class Network(nn.Module):
    def __init__(self, device='cpu', n_modes=44, n_frames=5, pixel_size=0.042, \
        telescope_diameter=150.0, central_obscuration=0.0, wavelength=8000.0, basis_for_wavefront='zernike', npix_image=128):
        
        super(Network, self).__init__()

        self.n_modes = n_modes
        self.n_frames = n_frames
        self.pixel_size = pixel_size
        self.telescope_diameter = telescope_diameter
        self.central_obscuration = central_obscuration
        self.wavelength = wavelength
        self.npix_image = npix_image
        self.basis_for_wavefront = basis_for_wavefront
        self.device = device

        print(f"Wavelength : {self.wavelength} A")
        print(f"Diameter : {self.telescope_diameter} cm")
        print(f"Central obscuration : {self.central_obscuration} cm")
        print(f"Pixel size : {self.pixel_size} arcsec")

        self.overfill = util.psf_scale(self.wavelength, self.telescope_diameter, self.pixel_size)                
        if (self.overfill < 1.0):
            raise Exception(f"The pixel size is not small enough to model a telescope with D={self.telescope_diameter} cm")
            
        # Compute telescope aperture
        pupil = util.aperture(npix=self.npix_image, cent_obs = self.central_obscuration / self.telescope_diameter, spider=0, overfill=self.overfill)
        pupil = torch.tensor(pupil.astype('float32'))
            
            # Define all KL modes
        if (self.basis_for_wavefront == 'zernike'):
            print("Computing Zernike modes...")
            Z_machine = zern.ZernikeNaive(mask=[])
            x = np.linspace(-1, 1, self.npix_image)
            xx, yy = np.meshgrid(x, x)
            rho = self.overfill * np.sqrt(xx ** 2 + yy ** 2)
            theta = np.arctan2(yy, xx)
            aperture_mask = rho <= 1.0

            basis = np.zeros((self.n_modes, self.npix_image, self.npix_image))
            
            # Precompute all Zernike modes except for piston
            for j in range(self.n_modes):
                n, m = zern.zernIndex(j+2)
                Z = Z_machine.Z_nm(n, m, rho, theta, True, 'Jacobi')
                basis[j,:,:] = Z * aperture_mask

        if (self.basis_for_wavefront == 'kl'):
            print("Computing KL modes...")
            kl = kl_modes.KL()
            basis = kl.precalculate_covariance(npix_image = self.npix_image, n_modes_max = self.n_modes, first_noll = 1, overfill=self.overfill)

        zeros = torch.zeros((self.npix_image, self.npix_image, 1), dtype=torch.float32)

        self.register_buffer('zeros', zeros)
        self.register_buffer('pupil', pupil)
        self.register_buffer('basis', torch.tensor(basis.astype('float32')))

        self.modalnet = Recurrentnet(in_planes=1, device=self.device, n_modes=self.n_modes, n_frames=self.n_frames, npix_image=self.npix_image, n=16, n_lstm=256).to(self.device)
        self.modalnet.weights_init()

    def compute_psfs(self, coeff):
        """Compute the PSFs and their Fourier transform from a set of modes
        
        Args:
            wavefront_focused ([type]): wavefront of the focused image
            illum ([type]): pupil aperture
            diversity ([type]): diversity for this specific images
        
        """
        
        # Compute real and imaginary parts of the pupil
        wavefront = torch.einsum('ij,jkl->ikl', coeff, self.basis)
        
        tmp1 = torch.unsqueeze(torch.cos(wavefront) * self.pupil[None, :, :], -1)
        tmp2 = torch.unsqueeze(torch.sin(wavefront) * self.pupil[None, :, :], -1)        

        # Compute complex phase
        phase = torch.cat([tmp1, tmp2], -1)

        # Compute FFT of the pupil function and compute autocorrelation
        ft = torch.ifft(phase, 2)
        psf = complex_multiply_astar_b(ft, ft)[..., 0]
        
        # Normalize PSF and transform to pytorch-complex
        tmp = torch.unsqueeze(psf / torch.sum(psf, [1, 2])[:, None, None], -1)

        # Set imaginary part to zero
        psf = torch.cat([tmp, self.zeros.expand(tmp.size(0), self.npix_image, self.npix_image, 1)], -1)

        # Compute Fourier transform of PSF for later convolutions
        psf_ft = torch.fft(psf, 2)

        return psf, psf_ft, wavefront

    def loss_and_wiener_filter(self, im_ft, psf_ft, variance):
        """Compute MOMFBD loss function and the estimated deconvolved image. See Michiel van Noorts and Mats Löfdahl papers
        
        Args:
            focused_ft (tensor): FFT of the focused images
            defocused_ft (tensor): FFT of the defocused images
            psf_focused_ft (tensor): FFT of the focused PSF
            psf_defocused_ft (tensor): FFT of the defocused PSF
        
        
        """

        # D = burst_ft
        # S = psf_ft
                
        # Compute S* x D
        S_star_D = complex_multiply_astar_b(psf_ft, im_ft)
        
        # Compute D* x S
        D_star_S = complex_multiply_astar_b(im_ft, psf_ft)
        
        # Compute modulus of S : |S|^2 = S* x S
        modulus_S = complex_multiply_astar_b(psf_ft, psf_ft)
        
        # Compute modulus of D : |D|^2 = D* x D
        modulus_D = complex_multiply_astar_b(im_ft, im_ft)
        
        # Compute modulus of the product between D^* and S summed for all frames
        sum_D_star_S = torch.sum(D_star_S, dim=1)
        modulus_D_star_S = complex_multiply_astar_b(sum_D_star_S, sum_D_star_S)

        # Wiener filter estimation of the image
        denominator = torch.sum(modulus_S, dim=1)
        # Q[..., 0] += 1e-10
        numerator = torch.sum(S_star_D, dim=1)

        # Loss function
        tmp = torch.sum(modulus_D, dim=1)

        loss = tmp[..., 0] - modulus_D_star_S[..., 0] / (variance[:, None, None, None] + denominator[..., 0])

        # This normalization is here because we use non-normalized FFTs, which
        # lack a sqrt(Nx*Ny). It is squared because the loss function has
        # squared FFTs
        loss_mn = torch.mean(loss) / (self.npix_image**2)

        return numerator, denominator, loss_mn

    def forward(self, images, images_ft, variance):
        
        coeff = self.modalnet(images)

        tmp = coeff.view(-1, self.n_frames, self.n_modes)

        # Force zero tip-tilt on average
        avg = torch.mean(tmp, dim=1, keepdim=True)
        avg[:, :, 2:] = 0.0

        avg = avg.expand(tmp.size(0), tmp.size(1), tmp.size(2)).reshape(-1, self.n_modes)

        psf, psf_ft, wavefront = self.compute_psfs(coeff - avg)

        psf_ft = psf_ft.view(-1, self.n_frames, self.npix_image, self.npix_image, 2)
        numerator, denominator, loss = self.loss_and_wiener_filter(images_ft, psf_ft, variance)
        
        return coeff - avg, numerator, denominator, psf, psf_ft, loss