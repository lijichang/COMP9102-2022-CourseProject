# COMP9102-2022-CourseProject

This is an implementation on Python 3.8  to analyze the performance differences among various algorithms involved in dimensionality reduction (DR), namely t-SNE, UMAP, PaCMAP and h-NNE.

This code is submitted as a course project of the PHD course COMP9102-2022 at HKUCS.

## Installation

You would require the following packages to fully use this code on your machine according to [installation](#installation.txt) file:

- scikit-learn
- numpy
- numba
- torchvision
- umap-learn
- pacmap
- hnne
- annoy
- tqdm
- p_tqdm

You can use pip to install the above packages from PyPI. It will automatically install the dependencies for you:

```
pip install scikit-learn==0.24.1
pip install numpy==1.19
pip install numba==0.51.0
pip install torchvision==0.2.1
pip install -U umap-learn==0.5.1
pip install pacmap==0.6.3
pip install hnne
pip install annoy
pip install -U tqdm==4.45.0
pip install -U p_tqdm==1.2
```


## Usage
### Evaluation

To evaluate the differences among various DR performance on three standard datasets, namely MNIST, SVHN and CIFAR-10, you can run a command as follows,

```
python evaluation.py
```
After the code running, you can obtain a csv file, where you can find the results of `KNN Accuracy`, `SVM Accuracy`, `Random Triplet Accuracy`, `Centroid Triplet Accuracy` and `running time`. 

### Visualization

To visualize the features obtained from various DR performance on three standard datasets, you can run a command as follows,
```
python visualization.py
```

## Acknowledgement

Thanks to [PaCMAP](https://github.com/YingfanWang/PaCMAP). We obtain great inspiration and reference from this code to implement our work.


