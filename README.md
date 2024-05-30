# Integrating Global Sparse Attention into ResNet

## Global Sparse Attention

Attention由于其二次复杂度而难以用在图像领域，vit采用将图像分成16*16个patches的形式，但这样在较大图片上仍然有不小的复杂度，swin transformer采用只在patch内部进行attention的操作但如此又回到了CNN失去了attention的全局属性

由我们日常的经验可以知道，将图片进行压缩不太会影响我们对一个图片中具有的物体的判断，于是我们可以在图片打成patches后对处于patch中相同位置的像素进行拼接，之后再对每一个分块进行attention的操作

<img src="./../pics/Snipaste_2024-04-24_10-07-01.png" style="zoom: 50%;" />

接着我们只需要再对attention操作后的分块出现进行拼接，这样就可以获得一张和原图相同大小的图

<img src="./../pics/Snipaste_2024-04-24_10-32-22.png" style="zoom: 50%;" />

后面我们还可以考虑对拼接后的图片在原位置进行attention操作，希望在这样稀疏的注意力机制后可以再通过这一步操作获得全局的特征信息

<img src="./../pics/Snipaste_2024-04-24_10-37-57.png" style="zoom:50%;" />



## Global Sparse Attention代码实现

在代码方面可以通过对fold后的矩阵进行permute巧妙的实现

```python
image = torch.arange(0, 32).reshape(2, 1, 4, 4).float()
spliter = nn.Unfold(kernel_size=2, stride=2)
sparse_image = spliter(image) 
sparse_image # shape: (batch, channels * kernel_size^2, num_patches)
```

<pre>tensor([[[ 0,  1,  2,  3]],
        [[ 4,  5,  6,  7]],
        [[ 8,  9, 10, 11]],
        [[12, 13, 14, 15]],
        <br4>
        [[16, 17, 18, 19]],
        [[20, 21, 22, 23]],
        [[24, 25, 26, 27]],
        [[28, 29, 30, 31]]])</pre>

```python
sparse_image = sparse_image.permute(1, 0, 2) # shape: (channels * kernel_size^2, batch, num_patches)
sparse_image # 此时就是以我们希望的分块方式构成矩阵了
```

<pre>
    tensor([[[ 0.,  2.,  8., 10.],
         [16., 18., 24., 26.]],
<br4>
        [[ 1.,  3.,  9., 11.],
         [17., 19., 25., 27.]],
<br4>
        [[ 4.,  6., 12., 14.],
         [20., 22., 28., 30.]],
<br4>
        [[ 5.,  7., 13., 15.],
         [21., 23., 29., 31.]]])
</pre>

```python
image_size = 4
channel_size = 1
patch_size = 2
num_heads = 2
# 我们只要让MultiheadAttention的条件满足我们构成矩阵的最后一个维度便可以进行运算了
across_patches_attn = nn.MultiheadAttention(
    embed_dim=image_size**2 // patch_size**2, num_heads=num_heads
)
global_attn, _ = across_patches_attn(sparse_image, sparse_image, sparse_image)
global_attn # shape: (num_patches, batch, embed_dim)
```

<pre>tensor([[[-1.1977e+00, -6.2631e-01,  5.7386e-02,  4.3052e-01],
         [-6.9679e+00,  4.3290e-01,  6.2930e+00, -3.4438e+00]],
<br4>
        [[-1.1995e+00, -6.6403e-01,  6.7582e-03,  4.5862e-01],
         [-6.9679e+00,  4.3289e-01,  6.2930e+00, -3.4438e+00]],
<br4>
        [[-1.2014e+00, -6.8125e-01, -1.5452e-02,  4.7086e-01],
         [-6.9679e+00,  4.3289e-01,  6.2930e+00, -3.4438e+00]],
<br4>
        [[-1.2015e+00, -6.8159e-01, -1.5891e-02,  4.7110e-01],
         [-6.9679e+00,  4.3289e-01,  6.2930e+00, -3.4438e+00]]],
       grad_fn=&ltViewBackward0>)</pre>

```python
patches = sparse_image.permute(2, 0, 1) # shape: (num_patches, batch, channels * kernel_size^2)
patches # 此时便是以patch的形式
```

<pre>
    tensor([[[ 0.,  1.,  4.,  5.],
         [16., 17., 20., 21.]],
<br4>
        [[ 2.,  3.,  6.,  7.],
         [18., 19., 22., 23.]],
<br4>
        [[ 8.,  9., 12., 13.],
         [24., 25., 28., 29.]],
<br4>
        [[10., 11., 14., 15.],
         [26., 27., 30., 31.]]])
</pre>

```python
in_patch_attn = nn.MultiheadAttention(
        embed_dim=channel_size * patch_size**2, num_heads=num_heads
)
in_patch_attn, _ = self.in_patch_attn(patches, patches, patches)
in_patch_attn
```

<pre>
    tensor([[[-1.2815,  2.2133, -1.9065,  1.8900],
         [-3.6350,  8.9643, -8.9213,  5.3031]],
<br4>
        [[-1.2819,  2.2303, -1.8811,  1.9157],
         [-3.6352,  8.9724, -8.9090,  5.3154]],
<br4>
        [[-1.4521,  3.0772, -1.7851,  2.7317],
         [-3.6323,  8.9773, -8.8780,  5.3322]],
<br4>
        [[-1.5338,  3.4758, -1.7518,  3.1110],
         [-3.6308,  8.9759, -8.8692,  5.3345]]], grad_fn=&ltViewBackward0>)
</pre>

> 假设如果一张图的大小为c * h * w，我们选取的patch的大小为k,那么这样的一套attention操作下来，相当于对一段长度为h * w / k * k 和一段长度为 c * k * k的句子进行attention操作
>
> 对于较大图片我们可以再分块后重复上述操作（毕竟对于较大图片我们可以更大程度的对图片进行缩放），这样如果选取合适的patch的大小可以把attention的复杂度控制在合理的范围

由于global sparse attention不改变输入图像的形状因此我们可以很容易的将其加入到其他模型中, 于是我尝试的将其加入了到了resnet中，并在cifar10上进行了测试，测试集准确度可以达到90%
