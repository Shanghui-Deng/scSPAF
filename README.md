## scSPAF: Cell Similarity Purified  Adaptive Fusion Network for Single-Cell Multi-Omics Clustering


## Datasets
Ma-2020, PBMC-3k, PBMC-10k, BMNC, GSE, and GSE100866 datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/17Q1rlOfqSKdVlIXqqs8oBC6BIpqaEt-o?usp=sharing).

## Pretrained Model 
The Pretrained Model can be downloaded from [Google Drive](https://drive.google.com/drive/folders/17Q1rlOfqSKdVlIXqqs8oBC6BIpqaEt-o?usp=sharing). It can also be obtained through code training.

## Requirement
- Pytorch --- 2.4.0
- Python --- 3.9.19
- Numpy --- 1.26.4
- Scipy --- 1.13.1
- Sklearn --- 1.5.2
- Munkres --- 1.1.4
- tqdm --- 4.66.5


## Usage

#### Clone this pro
```
git clone https://github.com/Shanghui-Deng/scSPAF.git
```
#### Code structure
- ```data_loader.py```: loads the dataset and contruct the cell graph
- ```opt.py```: defines parameters
- ```utils.py```: defines the utility functions
- ```encoder.py```: defines the AE and GAE
- ```scSPAF.py```: defines the architecture of the whole model
- ```main.py```: run the model

Train a new model:

````python
python main.py
````
