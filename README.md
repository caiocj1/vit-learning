# Vision Transformer Implementation
***

Implementation of the vision transformer from [1],
for self-educational purposes. Trains from scratch on ImageNet 1k. Uses
multiple GPUs with `nn.DataParallel`.

### Usage

Be sure to extract class-separated ImageNet into `inputs` folder following
https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh.

To launch training,
`python main.py -v <version_name>`

To track training, `tensorboard --logdir tb_logs`

### Useful repositories:

- https://github.com/lucidrains/vit-pytorch
- https://github.com/google-research/big_vision


### References

[1] Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv
preprint arXiv:2010.11929 (2020). https://arxiv.org/abs/2010.11929.
