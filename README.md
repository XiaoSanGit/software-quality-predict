# variational-autoencoder
use variational-autoencoder to predict the quality of software.
Given the features of every stages, throught latent variable which discribe the distribution, get the bugs/quality of the software finally. 
! if we can unify the feature of every stages, temporal model can be used.

INPUT: two parts,demands and codes is based on modules. 
demands is  (modules, features). codes is also (modules,features)
modules is feasible, features must be sure before training.
saved and read by .NPY file.

Feature extractor for VAE is considered.


This is code that goes along with [my post explaining the variational autoencoder.](http://kvfrans.com/variational-autoencoders-explained/)

Based off this [really helpful post](https://jmetzen.github.io/2015-11-27/vae.html)
