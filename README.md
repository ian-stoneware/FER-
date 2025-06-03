# FER-Classification Task

This project investigates the effect of various **learning rate (LR) scheduling strategies** on model performance for **Facial Expression Recognition (FER)**. All schedulers were initialized with a learning rate of **0.01**.

## 📊 LR Scheduling Strategies

| Scheduler                  | Description |
|---------------------------|-------------|
| **ReduceLROnPlateau**     | Reduces LR by a factor of **0.75** when validation accuracy stops improving for **5 epochs**, allowing adaptive fine-tuning. |
| **StepLR**                | Reduces LR by **50% every 20 epochs**, offering a simple, periodic schedule. |
| **OneCycleLR**            | Uses a cyclic policy: increases LR to a peak and then anneals, enabling **faster convergence**. |
| **CosineAnnealingWarmRestarts** | Applies cosine annealing with **warm restarts every 10 epochs** to help the optimizer escape local minima. |


## 🖥️ Environment

- **GPU**: NVIDIA RTX A5000  
- **CPU**: 12th Gen Intel Core i9-12900  
- **Python**: 3.9  
- **PyTorch**: 2.6.0+cu118  
