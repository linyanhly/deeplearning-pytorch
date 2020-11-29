### 实现问题

* 是否需要在\__init\__中将所有层全部定义出来，如果需要，定义到哪种程度
* 在实现的过程中，如何结合resnet网络结构，将resnet结构作为downsample的过程
* 如何crop，在这里我采用了padding，因此与原论文不符，说明了这个问题，因此如何crop并且与上采样后的层进行cat
* 在这里cat可以采用torch中的函数torch.cat
* 需要简化UNet的实现，因此需要定义bottlelayers，但是需要定义到什么程度是一个问题，或者说是如何定义，是必须采用class类吗