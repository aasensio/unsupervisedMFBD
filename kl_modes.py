import numpy as np
import zern
import matplotlib.pyplot as pl
from tqdm import tqdm
import scipy.special as sp

def _even(x):
    return x%2 == 0

def _zernike_parity( j, jp):
    return _even(j-jp)

class KL(object):

    def __init__(self):
        pass
    # tmp = np.load('kl/kl_data.npy')
        # self.noll_KL = tmp[:,0].astype('int')
        # self.noll_Z = tmp[:,1].astype('int')
        # self.cz = tmp[:,2]

    def precalculate(self, npix_image, first_noll=1):
        """
        Precalculate KL modes. We skip the first mode, which is just the
        aperture. The second and third mode (first and second one in the return)
        are tip-tilt using standard Zernike modes. The rest are KL modes
        obtained by the diagonalization of the Kolmogorov Noll's matrix
        
        Parameters
        ----------
        npix_image : int
            Number of pixels on the pupil plane
        
        Returns
        -------
        float array
            KL modes
        """
        
        self.npix_image = npix_image

        print("Computing KL modes...")
        Z_machine = zern.ZernikeNaive(mask=[])
        x = np.linspace(-1, 1, npix_image)
        xx, yy = np.meshgrid(x, x)
        rho = np.sqrt(xx ** 2 + yy ** 2)
        theta = np.arctan2(yy, xx)
        aperture_mask = rho <= 1.0

        self.KL = np.zeros((self.n_modes_max, self.npix_image, self.npix_image))
        
        for mode in (range(self.n_modes_max)):

            j = mode + first_noll
            
            # if (i <= 2):
            #     n, m = zern.zernIndex(i + first_noll)
            #     Z = Z_machine.Z_nm(n, m, rho, theta, True, 'Jacobi')
            #     self.KL[i,:,:] = Z * aperture_mask

            # else:

            indx = np.where(self.noll_KL == j)[0]

            tmp = vh[mode,:]
            print(mode, self.cz[indx], tmp[np.abs(tmp) > 0])
            
            for i in range(len(indx)):
                jz = self.noll_Z[indx[i]]
                cz = self.cz[indx[i]]
                n, m = zern.zernIndex(jz)
                Z = Z_machine.Z_nm(n, m, rho, theta, True, 'Jacobi')
                self.KL[mode,:,:] += cz * Z * aperture_mask

        return self.KL

    def precalculate_covariance(self, npix_image, n_modes_max, first_noll=1, overfill=1.0):
        """
        Precalculate KL modes. We skip the first mode, which is just the
        aperture. The second and third mode (first and second one in the return)
        are tip-tilt using standard Zernike modes. The rest are KL modes
        obtained by the diagonalization of the Kolmogorov Noll's matrix
        
        Parameters
        ----------
        npix_image : int
            Number of pixels on the pupil plane
        n_modes_max : int
            Maximum number of nodes to consider
        first_noll : int
            First Noll index to consider. j=1 is the aperture. j=2/3 are the tip-tilts
        
        Returns
        -------
        float array
            KL modes
        """

        self.npix_image = npix_image
        self.first_noll = first_noll - 1
        self.n_modes_max = n_modes_max + first_noll

        print("Computing Kolmogorov covariance...")
        covariance = np.zeros((self.n_modes_max, self.n_modes_max))
        for j in range(self.n_modes_max):
            n, m = zern.zernIndex(j+1)

            for jpr in range(self.n_modes_max):
                npr, mpr = zern.zernIndex(jpr+1)
                
                deltaz = (m == mpr) and (_zernike_parity(j, jpr) or m == 0)
                
                if (deltaz):                
                    phase = (-1.0)**(0.5*(n+npr-2*m))
                    t1 = np.sqrt((n+1)*(npr+1)) 
                    t2 = sp.gamma(14./3.0) * sp.gamma(11./6.0)**2 * (24.0/5.0*sp.gamma(6.0/5.0))**(5.0/6.0) / (2.0*np.pi**2)

                    Kzz = t2 * t1 * phase
                    
                    t1 = sp.gamma(0.5*(n+npr-5.0/3.0))
                    t2 = sp.gamma(0.5*(n-npr+17.0/3.0)) * sp.gamma(0.5*(npr-n+17.0/3.0)) * sp.gamma(0.5*(n+npr+23.0/3.0))
                    covariance[j,jpr] = Kzz * t1 / t2

        covariance[0,:] = 0.0
        covariance[:,0] = 0.0
        covariance[0,0] = 1.0
        
        print("Diagonalizing Kolmogorov covariance...")
        u, s, vh = np.linalg.svd(covariance)

        vh[np.abs(vh) < 1e-10] = 0.0

        print("Computing KL modes...")
        Z_machine = zern.ZernikeNaive(mask=[])
        x = np.linspace(-1, 1, npix_image)
        xx, yy = np.meshgrid(x, x)
        rho = overfill * np.sqrt(xx ** 2 + yy ** 2)
        theta = np.arctan2(yy, xx)
        aperture_mask = rho <= 1.0

        self.KL = np.zeros((self.n_modes_max, self.npix_image, self.npix_image))

        noll_Z = 1 + np.arange(self.n_modes_max)

        for mode in tqdm(range(self.n_modes_max)):
            
            cz = vh[mode,:]
            ind = np.where(cz != 0)[0]
            
            for i in range(len(ind)):
                jz = noll_Z[ind[i]]                
                coeff = cz[ind[i]]
                n, m = zern.zernIndex(jz)
                Z = Z_machine.Z_nm(n, m, rho, theta, True, 'Jacobi')
                self.KL[mode,:,:] += coeff * Z * aperture_mask
        
        self.KL = self.KL[self.first_noll+1:,:,:]
        
        return self.KL



if (__name__ == '__main__'):

    tmp = KL()
    
    tmp.precalculate_covariance(npix_image=128, n_modes_max=25, first_noll=3)

    f, ax = pl.subplots(nrows=4, ncols=4)
    for i in range(16):
        ax.flat[i].imshow(tmp.KL[i, :, :])
    pl.show()

    mat = np.zeros((20,20))
    for i in range(20): 
        for j in range(20): 
            mat[i,j] = np.sum(tmp.KL[i,:,:]*tmp.KL[j,:,:])

    #pl.imshow(np.log(np.abs(mat)))
    #pl.show()