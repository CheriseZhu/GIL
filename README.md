   Graph Geometry Interaction Learning (GIL) in PyTorch
==================================================

## 1. Overview

This repository is an implementation of our NeurIPS 2020 paper [Graph Geometry Interaction Learning](https://arxiv.org/abs/2010.12135) (```GIL```)  in PyTorch, based on [HGCN](https://arxiv.org/abs/1910.12933) implementation, including following baselines.

Schematic of GIL architecture.
![image](https://github.com/CheriseZhu/GIL/blob/master/model.png)
#### Shallow methods (```Shallow```)

  * Shallow Euclidean
  * [Shallow Hyperbolic](https://arxiv.org/pdf/1705.08039.pdf)
  
#### Neural Network (NN) methods 

  * Multi-Layer Perceptron (```MLP```)
  * [Hyperbolic Neural Networks](https://arxiv.org/pdf/1805.09112.pdf) (```HNN```) 
  
#### Graph Neural Network (GNN) methods 

  * [Graph Convolutional Neural Networks](https://arxiv.org/pdf/1609.02907.pdf) (```GCN```) 
  * [Graph Attention Networks](https://arxiv.org/pdf/1710.10903.pdf) (```GAT```) 
  * [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) (```SAGE```) 
  * [Simplifying graph convolutional networks](https://arxiv.org/abs/1902.07153) (```SGC```) 
  
#### Hyperbolic Graph Neural Network (HGNN) methods 
  * [Hyperbolic Graph Convolutions Networks](https://arxiv.org/abs/1910.12933) (```HGCN```) 
  * [Hyperbolic Graph Neural Networks](https://arxiv.org/abs/1910.12892) (```HGNN```) 
  * [Hyperbolic Graph Attention Networks](https://arxiv.org/abs/1912.03046) (```HGAT```) 
  
All models can be trained for 

  * Link prediction (```lp```)
  * Node classification (```nc```)

## 2. Setup
### 2.1 Requirements
python == 3.6.2<br>
torch == 1.1.0<br>
numpy == 1.16.4<br>
scipy == 1.3.0<br>
networkx == 2.3<br>
sage == 9.0<br>
geoopt ==0.0.1<br>
torch_scatter == 1.3.0<br>
torch_geometric == 1.3.0

### 2.2 Folders
  The ```data/``` folder contains five datasets: cora/citeseer/pubmed/disease/airport.<br>
  The ```layers/``` folder contains basic operations of euclidean ```layers``` and hyperbolic ```hyp_layers```.<br>
  The ```manifolds/``` folder contains basic operations of euclidean space and hyperbolic space.<br>
  The ```models/``` folder contains the implementation of baselines, which consist of encoder and decoder.<br>
  The ```utils/``` folder contains the basic utils for data/eval/train, besides, ```hyperbolicity``` is used for calculating the distribution of hyperbolicity.<br>

## 3. Usage

### 3.1 ```set_env.sh```

Before training, run 

```source set_env.sh```

This will create environment variables that are used in the code. 

### 3.2  ```train.py```

We provide examples of training commands used to train GIL and other baselines for link prediction and node classification, under the same random seed 1234 for reproducibility purposes. 

#### Link prediction for GIL

  * Disease GIL (Test ROC-AUC: 100.00):

```python train.py --task lp --dataset disease_lp --model GIL --dropout 0 --weight-decay 0 --manifold PoincareBall --normalize-feats 0 --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1```

  * Airport GIL (Test ROC-AUC: 98.78):

```python train.py --task lp --dataset airport --model GIL --dropout 0 --weight-decay 0.001 --manifold PoincareBall --normalize-feats 0 --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1```

  * Citeseer GIL (Test ROC-AUC: 100.00):

```python train.py --task lp --dataset citeseer --model GIL --dropout 0 --weight-decay 0.0005 --manifold PoincareBall --lr 0.01 --dim 16 --num-layers 1 --act relu --bias 1 --normalize-feats 0```

  * Cora GIL (Test ROC-AUC: 99.20):

```python train.py --task lp --dataset cora --model GIL --dropout 0.1 --weight-decay 0.005 --manifold PoincareBall --lr 0.01 --dim 16 --num-layers 1 --act relu --bias 1 --normalize-feats 0```

  * Pubmed GIL (Test ROC-AUC: 97.69):
  
```python train.py --task lp --dataset pubmed --model GIL --dropout 0.1 --weight-decay 0.0001 --manifold PoincareBall --lr 0.01 --dim 16 --num-layers 1 --act relu --bias 1```

#### Link prediction for other baselines

  * Disease HGCN (Test ROC-AUC: 86.40):

```python train.py --task lp --dataset disease_lp --model HGCN --dropout 0 --weight-decay 0 --manifold PoincareBall --normalize-feats 0 --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1```

  * Disease HGAT (Test ROC-AUC: 84.15):

```python train.py --task lp --dataset disease_lp --model HGAT --dropout 0 --weight-decay 0.0001 --manifold PoincareBall --normalize-feats 0 --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1```

  * Airport HGCN (Test ROC-AUC: 97.59):

```python train.py --task lp --dataset airport --model HGCN --dropout 0 --weight-decay 0 --manifold PoincareBall --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1```

  * Airport HGNN (Test ROC-AUC: 96.51):

```python train.py --task lp --dataset airport --model HGNN --dropout 0 --weight-decay 0 --manifold PoincareBall --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1```

  * Airport HGAT (Test ROC-AUC: 97.95):

```python train.py --task lp --dataset airport --model HGNN --dropout 0 --weight-decay 0 --manifold PoincareBall --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1```

  * Cora HGCN (Test ROC-AUC: 93.79):

```python train.py --task lp --dataset cora --model HGCN --dropout 0.1 --weight-decay 0.001 --manifold PoincareBall --lr 0.01 --dim 16 --num-layers 1 --act relu --bias 1 --normalize-feats 0```

  * Citeseer HGCN (Test ROC-AUC: 96.74):

```python train.py --task lp --dataset citeseer --model HGCN --dropout 0.5 --weight-decay 0.0001 --manifold PoincareBall --lr 0.01 --dim 16 --num-layers 1 --act relu --bias 1```

  * Citeseer HGNN (Test ROC-AUC: 93.36):

```python train.py --task lp --dataset citeseer --model HGNN --dropout 0.1 --weight-decay 0.0001 --manifold PoincareBall --lr 0.01 --dim 16 --num-layers 1 --act relu --bias 1```

#### Node classification for GIL

  * Disease GIL (Test accuracy: 92.52):

```python train.py --task nc --dataset disease_nc --model GIL --dropout 0.1 --weight-decay 0.001 --manifold PoincareBall --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1```

  * Airport GIL (Test accuracy: 91.22):

```python train.py --task nc --dataset airport --model GIL --dropout 0 --weight-decay 0.001 --manifold PoincareBall --normalize-feats 0 --lr 0.01 --dim 16 --num-layers 3 --act relu --bias 1```

  * Cora GIL (Test accuracy: 83.30):

```python train.py --task nc --dataset cora --model GIL --dropout 0.6 --weight-decay 0.001 --manifold PoincareBall --lr 0.01 --dim 16 --num-layers 2 --act elu --bias 1 --drop_h 0.9```

  * Pubmed GIL (Test accuracy: 78.40):

```python train.py --task nc --dataset pubmed --model GIL --dropout 0.6 --weight-decay 0.0005 --manifold PoincareBall --lr 0.01 --dim 16 --num-layers 2 --act elu --bias 1 --drop_h 0.9```

  * Citeseer GIL (Test accuracy: 71.50):

```python train.py --task nc --dataset citeseer --model GIL --dropout 0.5 --weight-decay 0.001 --manifold PoincareBall --lr 0.01 --dim 16 --num-layers 2 --act elu --bias 1 --drop_h 0.8```

#### Node classification for other baselines

  * Disease HGCN (Test accuracy: 91.34):

```python train.py --task nc --dataset disease_nc --model HGCN --dropout 0.2 --weight-decay 0.0005 --manifold PoincareBall --lr 0.01 --dim 16 --num-layers 3 --act relu --bias 1```

  * Airport HGCN (Test accuracy: 88.93):

```python train.py --task nc --dataset airport --model HGCN --dropout 0 --weight-decay 0 --manifold PoincareBall --normalize-feats 0 --lr 0.01 --dim 16 --num-layers 3 --act relu --bias 1```

  * Cora HGCN (Test accuracy: 78.60):

```python train.py --task nc --dataset cora --model HGCN --dropout 0.6 --weight-decay 0.0005 --manifold PoincareBall --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1```

  * Pubmed HGCN (Test accuracy: 77.00):

```python train.py --task nc --dataset pubmed --model HGCN --dropout 0.5 --weight-decay 0.0005 --manifold PoincareBall --lr 0.01 --dim 16 --num-layers 3 --act relu --bias 1```

  * Pubmed HGNN (Test accuracy: 74.50):

```python train.py --task nc --dataset pubmed --model HGNN --dropout 0.6 --weight-decay 0.0005 --manifold PoincareBall --lr 0.01 --dim 16 --num-layers 3 --act relu --bias 1```

  * Citeseer HGAT (Test accuracy: 69.50):
  
```python train.py --task nc --dataset citeseer --model HGAT --dropout 0.6 --weight-decay 0.001 --manifold PoincareBall --lr 0.01 --dim 16 --num-layers 3 --act elu --bias 1```

## Citation

If you find this code useful, please cite the following paper: 
```
@inproceedings{zhu2020GIL,
  author={Shichao Zhu and Shirui Pan and Chuan Zhou and Jia Wu and Yanan Cao and Bin Wang},
  title={Graph Geometry Interaction Learning},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```

## Some of the code was forked from the following repositories
 
 * [hgcn](https://github.com/HazyResearch/hgcn)
 * [geoopt](https://github.com/geoopt/geoopt)
 * [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric)
