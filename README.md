# Code for "GDAs-OT:  Prediction Method of Gene-Disease Associations Based on Optimal Transport for Identifying Genes Related to Immune-Related Adverse Events"

Authors: Ruhao Liu, Suixue Wang, Hang Yu, Peng Li, Qingchen Zhang.
If you have any questions, please contact Ruhao Liu (lrh@hainanu.edu.cn) or Qingchen Zhang (zhangqingchen@hainanu.edu.cn).

This directory contains implementations of GDAs-OT framework.

## Prepare material
You need to manually download the biomedical pre-trained language model PubMedBERT (https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext) and place it in the main directory of GDAs-OT. In addition, a dedicated conda environment needs to be created before running the code.

```shell
$ conda create -n GDAs-OT python=3.9
$ conda activate GDAs-OT
```

Then, ensure that the path leads to the GDAs-OT root directory and install requirements. Please carry out the following operations in sequence.

```shell
$ cd GDAs-OT
$ pip install torch==2.8.0
$ pip install -r requirements.txt
```

## Datasets
We provided processed data on genes, diseases, and gene-disease relationships. However, you still need to run the 'data_preprocessing/bert_embedding.py' script to obtain the node features for genes and diseases.

```shell
$ cd data_preprocessing
$ python bert_embedding.py
```

## Main program
run main.py

```shell
$ cd GDAs-OT
$ python main.py
```

you also can change hyper-parameters. For example:
```shell
$ python main.py --GCN_epochs 20 --MLP_epochs 15 --cost_type cosine --loss_type L2 --lr 0.001 --alpha 0.1 --beta 5.0 --margin 3.0 --sample_num 10000 --has_ot_matrix --model_controller gda_prediction 
```

