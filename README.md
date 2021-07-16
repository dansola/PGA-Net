# PGA-Net
Proposal Guided Attention Network

## About
Transformers are extremely powerful tools used in language models and are now being incorporated in computer vision problems.
Transformers offer a nice alternative to CNNs as they can immediately find global context in an image unlike convolutional kernels which require down-sampling.
A bottleneck of transformers however is the quadratic time efficiency of the attention mechanism making it unfeasible on images where
each pixel is treated as an element in the sequence.  We therefore must limit which pixels attend to one another if we wish to implement 
attention on images with the current available hardware.

This is a library I built to replicate the axial attention module in [Axial-DeepLab](https://arxiv.org/abs/2003.07853)
and improve upon it with something I call proposal guided attention (PGA).  PGA uses class agnostic image proposals 
created from frameworks like [DeepMask](https://arxiv.org/abs/1506.06204) or [SharpMask](https://arxiv.org/abs/1603.08695)
to guide pixel-wise attention.  Rather than forcing pixels of a certain row or column to attend to one another, PGA forces 
pixels of either all object or all background (according to the object proposal) to attend to one another.  

## Implementation
Images must first be passed through DeepMask or SharpMask to get object proposals ([official link](https://github.com/facebookresearch/deepmask), [pytorch link](https://github.com/foolwood/deepmask-pytorch)).
These proposals can then be converted to a proposal mask:
   ```
   cd src/datasets
   python build_proposal_masks.py --save-directory='(path to object proposals)' --n-proposals='(number of proposals to build mask with)'
   ```

These proposals are incorporated into a Dataset class where a random index dictionary is built for objects and background for each image.
Then PGA can be used on an image and its corresponding proposal dictionaries.

```python
from src.models.basic_pga.basic_pga_parts import BlockPGA
import torch

img = torch.rand(1,3,500,500) # test image
obj_dict, bg_dict = batch['obj_dict'], batch['bg_dict'] # proposal dictionaries from a dataloader
block = BlockPGA(channels=3, embedding_dims=10, img_shape=(500, 500))
out = block(img, obj_dict, bg_dict)
```

## Methodology
Pixels that reside within
an object proposal are selected randomly in order to assign pixels to attend to one another.  To do this, proposals are converted to 
dictionaries of indices to ensure that the image can be rebuilt after the attention module has re-weighted a tensor.

![alt text](https://github.com/dansola/PGA-Net/blob/main/images/rand_ind.png)

To be able to compare PGA with axial attention, the PGA module randomly selects *w* pixels for an attention tensor, and creates *h* attention tensors per head (*w* and *h* are the width and height of the image).
The random indices are created when the model is initialized and then they are fixed during training and saved with the model wights.

![alt text](https://github.com/dansola/PGA-Net/blob/main/images/pga.png)

A PGA block can be created similar to an axial block and can be stacked for image classification or segmentation.  
Due to inaccuracies in the object proposals, I opted to build an AxialPGA network where an image is passed to an axial block and 
a PGA block in parallel, their output is concatenated, and a 1x1 convolution weights the contribution of each.

![alt text](https://github.com/dansola/PGA-Net/blob/main/images/pga_block.png)

When comparing a 2 layer AxialPGA network (2 PGA block and 2 axial block) to a 4 layer Axial network for segmentation, the AxialPGA network
converges faster due to the guiding of the object proposals.  In this example the AxialPGA network only has 3.5k trainable parameters while the 
Axial network has 5k trainable parameters.

![alt text](https://github.com/dansola/PGA-Net/blob/main/images/converge.png)
 
## Acknowledgments

DeepMask proposals for testing were easily built with a pytorch implementation by [Qiang Wang](https://github.com/foolwood/deepmask-pytorch) where he included pre-trained weights.

Axial attention module was based on the implementation by [Phil Wang](https://github.com/lucidrains/axial-attention).