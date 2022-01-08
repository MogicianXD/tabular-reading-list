# Reading List of Tabular Data

## Survey

- [Deep Neural Networks and Tabular Data: A Survey](https://arxiv.org/abs/2110.01889) [Arxiv'21]

  > Vadim Borisov, Tobias Leemann, Kathrin Seßler, Johannes Haug, Martin Pawelczyk, Gjergji Kasneci
  > The University of Tübingen, Tübingen, Germany

- [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/pdf/2106.11959v2.pdf) [NIPS'21]

  > Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko 
  > Yandex, Russia

## Baselines

### Overview

Divided by task:

- *Data mining*: extracting information from a specific row with observed attributes, *i.e.*, **predicting labels with features**.

- *Data preparation*: addressing with missing or dirty value of rows, *i.e.*, **estimating features with or without label**.

| Model                                                        | Publication | Task                                                         | Key points                           | Codes                                                        |
| ------------------------------------------------------------ | ----------- | ------------------------------------------------------------ | ------------------------------------ | ------------------------------------------------------------ |
| [DANets](https://arxiv.org/pdf/2112.02962.pdf)               | AAAI'22     | classification, regression                                   |                                      |                                                              |
| [SAINT](https://arxiv.org/pdf/2106.01342.pdf)                | Arxiv'21    | classification                                               | Attention over both rows and columns |                                                              |
| [SCARF](https://arxiv.org/pdf/2106.15147.pdf)                | Arxiv'21    | classification                                               | Contrastive learning                 |                                                              |
| [Contrastive Mixup](https://arxiv.org/pdf/2108.12296.pdf)    | Arxiv'21    | classification                                               | SSL                                  |                                                              |
| [](https://arxiv.org/pdf/2110.13413.pdf)                     | Arxiv'21    | classification, regression                                   | GDBT                                 |                                                              |
| [TABBIE](https://arxiv.org/pdf/2105.02584.pdf)               | Arxiv'21    | column population, row population, and column type prediction | Corrupt cell detection               |                                                              |
| [SubTab](https://arxiv.org/pdf/2110.04361.pdf)               | NIPS'21     | classification                                               | Reconstructing data from its subset  |                                                              |
| [ARM-Net](https://arxiv.org/pdf/2107.01830.pdf)              | SIGMOD'21   | classification                                               | Attention                            | [Official](https://github.com/nusdbsystem/ARM-Net)           |
| [RPT](https://dl.acm.org/doi/pdf/10.14778/3457390.3457391)   | VLDB'21     | data preparation                                             | BERT and GPT                         |                                                              |
| [RIM](https://arxiv.org/pdf/2108.05252.pdf)                  | KDD'21      | classification, regression, ranking                          | Cross-row and cross-column           |                                                              |
| [TabularNet](https://arxiv.org/pdf/2106.03096.pdf)           | KDD'21      | classification                                               | Bi-GRU, GCN                          |                                                              |
| [Net-DNF](https://iclr.cc/virtual/2021/poster/2539)          | ICLR'21     | classification                                               | Hierarchical Modeling                | [Official](https://github.com/amramabutbul/DisjunctiveNormalFormNet) |
| [Boost-GNN](https://arxiv.org/pdf/2101.08543.pdf)            | ICLR'21     | classification                                               | GBDT                                 | [Official](https://github.com/nd7141/bgnn)                   |
| [TabGNN](https://arxiv.org/pdf/2108.09127.pdf)               | DLP-KDD'21  | classification, regression                                   | GNN                                  |                                                              |
| [DCN V2](https://arxiv.org/pdf/2008.13535v2.pdf)             | WWW'21      | classification, regression                                   | MLP                                  |                                                              |
| [TURL](https://dl.acm.org/doi/pdf/10.14778/3430915.3430921)  | VLDB'20     | learn task-agnostic contextualized representations           | Transformer                          |                                                              |
| [GRAPE](https://arxiv.org/pdf/2010.16418.pdf)                | NIPS'20     | feature imputation                                           | Edge-level prediction                |                                                              |
| [VIME](https://proceedings.neurips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html) | NIPS'20     | classification                                               | SSL                                  | [Official](https://github.com/vanderschaarlab/mlforhealthlabpub/tree/main/alg/vime) |
| [NON](https://arxiv.org/pdf/2005.10114.pdf)                  | SIGIR'20    | classification                                               |                                      |                                                              |
| [GrowNet](https://arxiv.org/pdf/2002.07971v2.pdf)            | Arxiv'20    | classification, regression                                   | Gradient boosting                    | [Official](https://github.com/sbadirli/GrowNet)              |
| [TabTransformer](https://arxiv.org/pdf/2012.06678.pdf)       | Arxiv'20    | classification                                               | Transformer                          |                                                              |
| [TabNet](https://arxiv.org/pdf/1908.07442v5.pdf)             | AAAI'21     | classification, regression                                   | Sequential attention                 | [Official](https://github.com/google-research/google-research/tree/master/tabnet) |
| [NODE](https://arxiv.org/pdf/1909.06312v2.pdf)               | ICLR'20     | classification, regression, ranking                          | Trees                                | [Official](https://github.com/Qwicen/node)                   |
| [AutoInt]()                                                  | CIKM'19     | classification                                               | Attention                            | [Official](https://github.com/DeepGraphLearning/RecommenderSystems) |
| [SuperTML](https://arxiv.org/pdf/1903.06246.pdf)             | CVPR'19     |                                                              | CNN                                  |                                                              |
| [CatBoost](https://arxiv.org/pdf/1706.09516v2.pdf)           | NIPS'18     |                                                              | GBDT                                 |                                                              |
| [XGBoost](https://dl.acm.org/doi/pdf/10.1145/2939672.2939785) | KDD'16      |                                                              | GBDT                                 |                                                              |

### Data Preparation

#### TABBIE

[TABBIE: Pretrained Representations of Tabular Data](https://arxiv.org/pdf/2105.02584.pdf)

> *Hiroshi Iida, Dung Thai, Varun Manjunatha, Mohit Iyyer*
> Sony, UMass Amherst, Adobe

A self-supervised pretraining approach trained exclusively on tables by asking the model to predict whether or not each cell in a table is real or corrupted.

#### RPT

[RPT: Relational Pre-trained Transformer Is Almost All You Need towards Democratizing Data Preparation](https://dl.acm.org/doi/pdf/10.14778/3457390.3457391) [VLDB'21]

> *Nan Tang, Ju Fan, Fangyi Li, Jianhong Tu, Xiaoyong Du, Guoliang Li, Sam Madden, Mourad Ouzzani*
> QCRI, HBKU, Qatar, Renmin University

#### GRAPE

[Handling Missing Data with Graph Representation Learning](https://arxiv.org/pdf/2010.16418.pdf) [NIPS'20]

> *Jiaxuan You, Xiaobai Ma, Daisy Yi Ding, Mykel Kochenderfer, Jure Leskovec*
> Stanford University

Feature imputation is formulated as an edge-level prediction task and the label prediction as a node-level prediction task in a GNN.

### Data Mining

#### DANets

[DANETs: Deep Abstract Networks for Tabular Data Classification and Regression](https://arxiv.org/pdf/2112.02962.pdf) [AAAI'22]

> *Jintai Chen, Kuanlun Liao, Yao Wan, Danny Z. Chen, Jian Wu*
> ZJU

Explicitly group correlative input features and generate higher-level features for semantics abstraction.

Baseline: XGBoost, CatBoost, gcForest, Net-DNF, TabNet, NODE, FCNN

#### SAINT

[SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training](https://arxiv.org/pdf/2106.01342.pdf)

> *Gowthami Somepalli, Micah Goldblum, Avi Schwarzschild, C. Bayan Bruss, Tom Goldstein*
> University of Maryland, College Park

Attention over both rows and columns

Baseline: XGBoost, LightGBM, CatBoost, MLP w. DAE, VIME, TabNet w. MLM, TabTransf. w. RTD

#### SCARF

[SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption](https://arxiv.org/pdf/2106.15147.pdf)

> *Dara Bahri, Heinrich Jiang, Yi Tay, Donald Metzler*
> Google

Self-supervised contrastive learning

Baseline: Methods of label noise or data augmentation

#### Contrastive Mixup

[Contrastive Mixup: Self- and Semi-Supervised learning for Tabular Domain](https://arxiv.org/pdf/2108.12296.pdf)

> *Sajad Darabi, Shayan Fazeli, Ali Pazoki, Sriram Sankararaman, Majid Sarrafzadeh*
> UCLA

Baseline: CatBoost, Manifold Mixup, VIME

#### EBBS

[Convergent Boosted Smoothing for Modeling Graph Data with Tabular Node Features](https://arxiv.org/pdf/2110.13413.pdf) 

> *Jiuhai Chen, Jonas Mueller, Vassilis N. Ioannidis, Soji Adeshina, Yangkun Wang, Tom Goldstein, David Wipf*
> University of Maryland, Amazon

Baseline: GAT, GCN, AGNN, APPNP, CatBoost, Boost-GNN

#### ARM-Net

[ARM-Net: Adaptive relational modeling with multi-headgated attention network](https://arxiv.org/pdf/2107.01830.pdf) [SIGMOD'21]

> *Shaofeng Cai, Kaiping Zheng, Gang Chen, H. V. Jagadish, Beng Chin Ooi, Meihui Zhang*
> NUS, Zhejiang University, University of Michigan, Beijing Institute of Technology

Adaptive relational modeling with multi-headgated attention network

Baseline: FM models, AFM, HOFM, DCN, CIN, AFN, DNN, GCN, GAT, AFN+

#### SubTab

[SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning](https://arxiv.org/pdf/2110.04361.pdf) [NIPS'21]

> *Talip Ucar, Ehsan Hajiramezanali, Lindsay Edwards*
> Respiratory and Immunology, R&D, AstraZeneca

Reconstructing the data from the subset of its features rather than its corrupted version in an autoencoder setting can better capture its underlying latent representation.

Baseline: XGBoost, AE, DAE, CAE, VIME-self

#### RIM

[Retrieval & Interaction Machine for Tabular Data Prediction](https://arxiv.org/pdf/2108.05252.pdf) [KDD'21]

> *Jiarui Qin, Weinan Zhang, Rong Su, Zhirong Liu, Weiwen Liu, Ruiming Tang, Xiuqiang He, Yong Yu*
> SJTU, Huawei Noah’s Ark Lab

Fully exploits both cross-row and cross-column patterns.

Baseline: FM models, GBDT, IPNN, PIN, FGCNN

#### TabularNet

[TabularNet: A Neural Network Architecture for Understanding Semantic Structures of Tabular Data](https://arxiv.org/pdf/2108.05252.pdf)

> *Lun Du, Fei Gao, Xu Chen, Ran Jia, Junshan Wang, Jiang Zhang, Shi Han, Dongmei Zhang*
> MSRA, Beijing Normal University, Peking University

Use Bi-GRU and GCN to simultaneously extract spatial and relational information from tables.

Baseline: SVM, CART, FCNN-MT, TAPAS

#### Net-DNF

[Net-DNF: Effective Deep Modeling of Tabular Data](https://iclr.cc/virtual/2021/poster/2539)[ICLR'21]

> *Liran Katzir, Gal Elidan, Ran El-Yaniv*
> Google

Based on disjunctive normal form.

Baseline: XGboost, FCN

#### Boost-GNN

[Boost then Convolve: Gradient Boosting Meets Graph Neural Networks](https://arxiv.org/pdf/2101.08543.pdf) [ICLR'21]

> *Sergei Ivanov, Liudmila Prokhorenkova*
> Criteo AI Lab, Yandex; HSE University; MIPT

GNN on top decision trees from the GBDT algorithm

Baseline: CatBoost, LightGBM, GAT, GCN, AGNN, APPNP, FCNN, FCNN-GNN

#### TabGNN

[TabGNN: Multiplex Graph Neural Network for Tabular Data Prediction](https://arxiv.org/pdf/2108.09127.pdf) [DLP-KDD'21]

> *Xiawei Guo, Yuhan Quan, Huan Zhao, Quanming Yao, Yong Li, Weiwei Tu*
> 4paradigm, Tsinghua

#### DCN V2

[DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems](https://arxiv.org/pdf/2008.13535v2.pdf) [WWW'21]

> *Ruoxi Wang, Rakesh Shivanna, Derek Z. Cheng, Sagar Jain, Dong Lin, Lichan Hong, Ed H. Chi*
> Google

Consists of an MLP-like module and the feature crossing module (a combination of linear layers and multiplications).

Baseline: PNN, DeepFM, DLRM, DCN, xDeepFM, AutoInt, CrossNet

#### TURL

[TURL: table understanding through representation learning](https://dl.acm.org/doi/pdf/10.14778/3430915.3430921)

> *Xiang Deng, Huan Sun, Alyssa Lees, You Wu, Cong Yu*
> The Ohio State University, Google

A transformer with a "visibility mask" is applied to get contextualized representations. The "visibility mask" is constructed so (1) table caption and topic entity are visible to all components of the table, (2) entities and text content in the same row or the same column are visible to each other.

#### VIME

[VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain](https://proceedings.neurips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html) [NIPS'20]

> *Jinsung Yoon, Yao Zhang, James Jordon, Mihaela van der Schaar*
> Google Cloud AI, UCLA, Cambridge, Oxford

Baseline: XGBoost, DAE, Context Encoder, MixUp

#### NON

[Network On Network for Tabular Data Classification in Real-world Applications](https://arxiv.org/pdf/2005.10114.pdf) [SIGIR'20]

> *Yuanfei Luo, Hao Zhou, Weiwei Tu, Yuqiang Chen, Wenyuan Dai, Qiang Yang*
> 4Paradigm Inc.

Field-wise network, across field network and operation fusion network

#### GrowNet

[Gradient Boosting Neural Networks: GrowNet](https://arxiv.org/pdf/2002.07971v2.pdf)

> *Sarkhan Badirli, Xuanqing Liu, Zhengming Xing, Avradeep Bhowmik, Khoa Doan, Sathiya S. Keerthi*
> Purdue University, UCLA, Linkedin, Amazon

Gradient boosted weak MLPs.

Baseline: XGBoost, AdaNet

#### TabTransformer

[TabTransformer: Tabular Data Modeling Using Contextual Embeddings](https://arxiv.org/pdf/2012.06678.pdf) 

> *Xin Huang, Ashish Khetan, Milan Cvitkovic, Zohar Karnin*
> Amazon AWS

MLP, Sparse MLP, LR, GBDT, TabNet, VIB

#### TabNet

[TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/pdf/1908.07442v5.pdf) [AAAI'21]

> *Sercan O. Arik, Tomas Pfister*
> Google Cloud AI

A recurrent architecture that alternates dynamical reweighing of features and conventional feed-forward modules.

Baseline: CatBoost, XGBoost, LightGBM

#### NODE

[Neural oblivious decision ensembles for deep learning on tabular data](https://arxiv.org/pdf/1909.06312v2.pdf) [ICLR'20]

> *Sergei Popov, Stanislav Morozov, Artem Babenko* 
> Yandex, Russia

A differentiable ensemble of oblivious decision trees.

Baseline: CatBoost, XGBoost, FCNN, mGBDT, DeepForest

#### AutoInt

[AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/pdf/1810.11921v2.pdf) [CIKM'19]

> *Weiping Song, Chence Shi, Zhiping Xiao, Zhijian Duan, Yewen Xu, Ming Zhang, Jian Tang*
> Peking University

Transforms features to embeddings and applies a series of attention-based transformations to the embeddings.

Baseline: AFM, DeepCrossing, NFM, CrossNet, CIN, HOFM



## Dataset

### Classification

|                                                              | Area     | Task                                                         | #Row              | #Col | Type                         | Missing values | Citation              |
| ------------------------------------------------------------ | -------- | ------------------------------------------------------------ | ----------------- | ---- | ---------------------------- | -------------- | --------------------- |
| [Forest Cover Type](http://archive.ics.uci.edu/ml/datasets/Covertype) | Forest   | Classification of forest cover type from cartographic variables | 581012            | 54   | Categorical, Integer         | w/o            | TabNet                |
| [Poker Hand](http://archive.ics.uci.edu/ml/datasets/Poker+Hand) | Poker    | Classification of the poker hand from the raw suit and rank attributes of the cards. | 1025010           | 11   | Categorical, Integer         | w/o            | TabNet                |
| [KDD Census Income ](http://archive.ics.uci.edu/ml/datasets/Census-Income+%28KDD%29) | Census   | To predict whether income exceeds $50K/yr based on  demographic and employment related variables | 299285            | 40   | Categorical, Integer         | w              | TabNet, VIME          |
| [Mushroom](http://archive.ics.uci.edu/ml/datasets/Mushroom)  | Mushroom | Mushroom edibility prediction                                | 8124              | 22   | Categorical                  | w              | TabNet                |
| [Higgs](https://archive.ics.uci.edu/ml/datasets/HIGGS)       | Physical | To distinguish between a signal process which produces Higgs bosons and a background process which does not. | 11000000          | 28   | Real                         |                | TabNet, GrowNet, NODE |
| [Epsilon](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html) | CV       |                                                              | 400,000 / 100,000 | 2000 |                              | w/o            | NODE                  |
| [Click](https://www.kaggle.com/c/kddcup2012-track2) / KDD12  | Ads      | CTR                                                          |                   | 12   | Categorical, Integer, String |                | AutoInt, NODE         |
| [Avazu](https://www.kaggle.com/c/avazu-ctr-prediction)       | Ads      | CTR                                                          |                   | 24   | Categorical                  | w/o            | AutoInt, Boost-GNN    |
| [Criteo](https://www.kaggle.com/c/criteo-display-ad-challenge) | Ads      | CTR                                                          |                   | 27   | Categorical, Integer,        |                | AutoInt, DCN-V2       |
| [Kick](https://www.kaggle.com/c/DontGetKicked)               | Used car | To predict if the car purchased is bad                       |                   | 34   |                              | w              |                       |
| [MiniBooNe](https://archive.ics.uci.edu/ml/datasets/MiniBooNE+particle+identification) | Physical | To distinguish electron neutrinos  from muon neutrinos       | 130065            | 50   | Categorical                  |                |                       |
| MNIST                                                        | Image    | Classification of hand-written numbers                       |                   | 784  | Integer                      | w/o            | VIME                  |
| [OpenML CC18 benchmark](https://www.openml.org/s/99)         |          |                                                              |                   |      |                              |                | SCARF                 |

### Regression

|                                                              | Area   | Task                                                         | #Row           | #Col | Type                   | Missing values | Citation        |
| ------------------------------------------------------------ | ------ | ------------------------------------------------------------ | -------------- | ---- | ---------------------- | -------------- | --------------- |
| [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) | Store  | To forecast the store sales from static and time-varying features. |                | 15   | Categorical, Numerical |                | TabNet          |
| [YearPrediction](https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd) | Song   | To Predict  the release year                                 | 463715 / 51630 | 90   | Real                   |                | GrowNet, NODE   |
| [CT Slice Localization](http://archive.ics.uci.edu/ml/datasets/Relative+location+of+CT+slices+on+axial+axis) | Images | To retrieve the location of CT slices on axial axis.         | 53500          | 386  | Real                   | w/o            | GrowNet         |
| [MovieLen-1M](https://grouplens.org/datasets/movielens/)     | Movie  | To predict ratings                                           | 1 million      |      | Categorical, Numerical |                | AutoInt, DCN-V2 |

### Ranking

|                                                              | Area | Task | #Row | #Col | Type | Missing values | Citation      |
| ------------------------------------------------------------ | ---- | ---- | ---- | ---- | ---- | -------------- | ------------- |
| [Microsoft](https://www.microsoft.com/en-us/research/project/mslr/) |      |      |      |      |      |                | GrowNet, NODE |
| [Yahoo](https://webscope.sandbox.yahoo.com/catalog.php?datatype=c&guccounter=1) |      |      |      |      |      |                | GrowNet, NODE |



