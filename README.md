# Learning to do multiframe blind deconvolution unsupervisedly

[![github](https://img.shields.io/badge/GitHub-aasensio%2Fsicon-blue.svg?style=flat)](https://github.com/aasensio/unsupervisedMFBD)
[![license](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/aasensio/unsupervisedMFBD/blob/master/LICENSE)
[![ADS](https://img.shields.io/badge/ADS-arXiv200601438A-red.svg)](https://ui.adsabs.harvard.edu/abs/2020arXiv200601438A/abstract)
[![arxiv](https://img.shields.io/badge/arxiv-2006.01438-orange.svg?style=flat)](https://arxiv.org/abs/2006.01438)

Observation from ground based telescopes are affected by the presence of the 
Earth atmosphere, which severely perturbs them. The use of adaptive optics techniques
has allowed us to partly beat this limitation. However, image selection or
post-facto image reconstruction 
methods are routinely needed to reach the diffraction limit of telescopes. Deep learning has
been recently used to accelerate these image reconstructions.
Currently, these deep neural networks are trained with supervision, so that standard
deconvolution algorithms need to be applied a-priori to generate the training sets.

Our aim is to propose an unsupervised method which can then be trained simply with observations
and check it with data from the FastCam instrument.

We use a neural model composed of three neural networks that are trained end-to-end by 
leveraging the linear image formation theory to construct a physically-motivated loss function.

The analysis of the trained neural model shows that multiframe blind deconvolution can be trained 
self-supervisedly, i.e., using only observations. The output of the network are the corrected images
and also estimations of the instantaneous wavefronts. The network model is of the order of 1000
times faster than applying standard deconvolution based on optimization. With some work, the model
can bed used on real-time at the telescope.

## Training

This repository contains all the infrastructure needed to retrain the neural approach. However,
you will need to build a training set and do the needed modifications in the `train.py` file to
use your training set. You only need to provide bursts of images for the training since
no supervision is required. You will also need to adapt the sizes of the telescope primary
and secondary mirror, observing wavelength and pixel size in arcsec for the training to
proceed correctly.

We have tested with PyTorch 1.5 but it should work in all versions above 1.0.

### Dependencies
    numpy
    h5py
    torch
    tqdm
    argparse
    scipy

## Validation

This repository contains an example of an observation of sigOri with 200 frames and
the network trained for observations with the Nordic Optical Telescope (NOT) at 800 nm. The 
file `validation.py` shows how to apply the neural deconvolution to this example.

### Dependencies

    numpy
    matplotlib
    astropy
    torch
    tqdm
    skimage
    scipy
