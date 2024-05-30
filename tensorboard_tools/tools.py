import os
import shutil
import torch

import numpy as np
from matplotlib import pyplot as plt
from torch.nn import functional as F


def clear_tensorboard_runs(tensorboard_runs_dir: str) -> None:
    # 检查runs目录是否存在
    if os.path.exists(tensorboard_runs_dir):
        # 清除runs目录下的所有内容
        shutil.rmtree(tensorboard_runs_dir) # type: ignore
        print(f"Cleared all content in {tensorboard_runs_dir}")
    else:
        print(f"Directory {tensorboard_runs_dir} does not exist.")


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def images_to_probs(net, images):
    output = net(images)
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    preds, probs = images_to_probs(net, images)
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title(
            "{0}, {1:.1f}%\n(label: {2})".format(
                preds[idx],
                probs[idx] * 100.0,
                labels[idx],
            ),
            color=("green" if preds[idx] == labels[idx].item() else "red"),
        )
    return fig
