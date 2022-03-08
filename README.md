Novel Architecture of Parameterized Quantum Circuit for Graph Convolutional Network in Python 
====
The repository is the implementation of quantum convolutional networks (QGCN) based on quantum parameterized circuits, which is a quantum counterpart of [1]. QGCN Integrates the parameter-shift rule [2], which can use quantum circuits to find the gradient of tunable parameters.

This implementation makes use of the Cora dataset from [3].

## Requirements

  * Python  3.6 + 

## Usage

```python main.py```

## References

[1] [Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks, 2016](https://arxiv.org/abs/1609.02907)

[2] Crooks G E. Gradients of parameterized quantum gates using the parameter-shift rule and gate decomposition[J]. arXiv preprint arXiv:1905.13311, 2019.

[3] [Sen et al., Collective Classification in Network Data, AI Magazine 2008](http://linqs.cs.umd.edu/projects/projects/lbc/)

## Cite

Please cite our paper if you use this code in your own work:

http://arxiv.org/abs/2203.03251