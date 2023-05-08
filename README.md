# All_Expt_HAM10000-BACH
Different_Expt_on_HAM10000
This repository contains code to add pairflip noise, a kind of closed-set noise, to the HAM10000 dataset. With the "bkl" and "bcc" classes removed, the noisy dataset is trained and tested on clean and noisy dataset using GoogleNet, and MLPNet model.
Pairflip noise is a type of closed-set noise where the labels of pairs of images are randomly swapped. For example, if two images have labels "akiec" and "mel", the labels can be flipped to "mel" and "akiec". This type of noise is useful for simulating real-world scenarios where mistakes can happen in the labeling process.
To use the code in this repository, first download the HAM10000 dataset and preprocess it to drop the "bkl" and "bcc" classes. 
