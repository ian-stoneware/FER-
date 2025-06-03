# FER-Classification Task
In this work, we evaluated the impact of different LR scheduling strategies on model performance for FER. The initial LR for every scheduler is set to 0.01. We compared four common schedulers:
ReduceLROnPlateau: Reduces the LR by a factor of 0.75 when the test accuracy stops improving for 5 epochs, allowing adaptive fine-tuning.
StepLR: Reduces the LR by half every 20 epochs, providing a simple, periodic reduction.
OneCycleLR: Uses a cyclic LR policy that increases the LR to a maximum value before annealing it down, targeting faster convergence.
CosineAnnealingWarmRestarts: Applies a cosine annealing schedule with warm restarts every 10 epochs to help the optimizer escape local minima and saddle point.
## Environment
GPU: NVIDIA RTX A5000
CPU: 12th Gen Intel(R) Core(TM) i9-12900
Python: 3.9
PyTorch: 2.6.0+cu118
