# SIFT_Neural_Net

Explore the use of SIFT descriptors in place of pooling layers in traditional CNN architectures. Explore the idea that max-pooling layers discard translational and rotational relationships between low-level features while SIFT descriptors capture the spatial relationship and orientation. Visualize pre-FC activations per class with t-SNE (below).

<img src="https://github.com/jamessoole/SIFT_Neural_Net/blob/main/images_visualize_final/tmp_vis_sift_1chann_15comp_1000iter_40epoch.jpg?raw=true" width="600">

Based on  “Exploiting SIFT Descriptor for Rotation Invariant Convolutional Neural Network” by Kumar et. al: https://arxiv.org/ftp/arxiv/papers/1904/1904.00197.pdf

Original SIFT Paper: https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf \
SIFT Descriptor implementation in PyTorch: https://github.com/ducha-aiki/pytorch-sift

