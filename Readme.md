# About

This is a deep learning framework for generating syntax trees of artificial mathmatical formulars which look
similar to those from a given image dataset. For the training of a generating recurrent neural network (RNN) 
I use an adversarial, following Goodfellow et al. (2014). A convolutional neural network (CNN) has been used as 
discriminator. The overall approach borrows heavily from SeqGAN (Yu et al., 2017). 

The generator iteratively generates tokens based on the subsequence generated so far. The sequence is interpreted as a
syntax tree in post-fix notation. Saturated trees are then translated to LaTeX and compiled to images. The output of
the discriminator for such images serve as reward for the generator. The generators parameters are trained with 
policy gradient. Unlike SeqGAN I don't employ a pre-training, since the image training data can not be used for maximum-likelihood 
estimation on sequences.

I used a dataset of formulars extracted from scientific papers downloaded from 
https://arxiv.org/. Similar datasets are available at https://whadup.github.io/EquationLearning/. The LaTeX formulars
have been compiled to an image format. 



