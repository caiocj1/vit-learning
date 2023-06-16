# Vision Transformer Implementation
***

Implementation of the vision transformer from [1] in PyTorch,
for self-educational purposes. Trains from scratch. Uses
multiple GPUs with `nn.DataParallel`.

### Usage

To launch training,
`python main.py -v <version_name> -i <path_to_dataset>`.

Make sure the path given has `train` and `val` folders with images separated by class.

To track training, `tensorboard --logdir tb_logs`.

### Useful repositories:

- https://github.com/lucidrains/vit-pytorch
- https://github.com/google-research/big_vision


### References

[1] Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv
preprint arXiv:2010.11929 (2020). https://arxiv.org/abs/2010.11929.
