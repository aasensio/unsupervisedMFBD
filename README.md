# unsupervisedMFBD
Learning to do multiframe blind deconvolution

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
