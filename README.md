# Fast as CHITA: Neural Network Pruning with Combinatorial Optimization

This is the offical repo of the  paper **DCADP**

## Requirements
This code has been tested with Python 3.7 and the following packages:
```
numba==0.56.4
numpy==1.21.6
scikit_learn==1.0.2
torch==1.12.1+cu113
torchvision==0.13.1+cu113
```

## Pruned models
We provide checkpoints for our best pruned models, obtained with the gradual pruning procedure described in the paper.

### MobileNetV1
|Sparsity|Checkpoint|
|--------|----------|



## Structure of the repo
Scripts to run the algorithms are located in `scripts/`. The current code supports the following architectures (datasets): MLPNet (MNIST), ResNet20 (Cifar10), MobileNetV1 (Imagenet) and ResNet50 (Imagenet). Adding new models can be done through `model_factory` function in `utils/main_utils.py`. 


## Citing DCADP
If you find DCADP useful in your research, please consider citing the following paper.

```





