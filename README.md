# SVM-classifier
Soft-margin SVM from scratch in Python/NumPy for classifying handwritten digits (even vs. odd) on a MNIST-style dataset.

This project implements a soft-margin support vector machine (SVM) from scratch in Python/NumPy to classify handwritten digits from a MNIST-style dataset. Each 28×28 grayscale image is flattened into a 784-dimensional feature vector and recoded into a binary label (even vs. odd). The model optimises the hinge-loss objective with L2 regularisation using (sub)gradient descent, and is evaluated on a held-out test set using classification accuracy and confusion matrices.
