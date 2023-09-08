# [CIKM2023] MPerformer: An SE(3) Transformer-based Molecular Perceptron
This is the official implementation of the full paper "MPerformer: An SE(3) Transformer-based Molecular Perceptron" [[Paper](https:)]

MPerformer is a universal learning-based molecular perception method to construct 3D molecules with complete chemical information purely based on molecular 3D atom clouds.
<p align="center"><img src="figures/Overview.png" width=80%></p>
<p align="center"><b>An illustration of MPerformer and its learning paradigm. 
    Given an atom cloud, MPerformer derives its atom-level and pair-level representations and predicts a 3D molecule with complete chemical information. 
    During pretraining, we connect MPerformer with a decoder, reconstructing the atom cloud from its noisy versions.</b></p>
