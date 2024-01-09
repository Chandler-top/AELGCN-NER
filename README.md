# AELGCN-NER
Pytorch implementation of Attention and Edge-Label Guided Graph Convolutional Networks for Named Entity Recognition

EMNLP 2022 [Edge-Label Guided Graph Convolutional Networks for Named Entity Recognition](https://aclanthology.org/2022.emnlp-main.436/)
# Model Architecture

# Requirement
Python 3.7

Pytorch 1.4.0

Transformers 3.3.1

CUDA 10.1, 10.2

# Running
Firstly, download the embedding files: [glove.6B.100d.txt](https://nlp.stanford.edu/projects/glove/) , [cc.ca.300.vec](https://fasttext.cc/docs/en/crawl-vectors.html), [cc.es.300.vec](https://fasttext.cc/docs/en/crawl-vectors.html), [cc.zh.300.vec](https://fasttext.cc/docs/en/crawl-vectors.html), and put the files in the data folder.

By default, the model eval our saved model (without BERT) on SemEval 2010 Task 1 Spanish dataset.

    python main.py  
To train the model with other datasets:

    python main.py --mode=train --dataset=ontonotes --embedding_file=glove.6B.100d.txt

For more detailed usage, please refer to the [SynLSTM-for-NER project](https://github.com/xuuuluuu/SynLSTM-for-NER?tab=readme-ov-file#better-feature-integration-for-named-entity-recognition)

# Related Repo
The code are created based on the codes of the paper ["Dependency-Guided LSTM-CRF Model for Named Entity Recognition"](https://github.com/allanj/ner_with_dependency), EMNLP 2019
["Better Feature Integration for Named Entity Recognition"](https://github.com/xuuuluuu/SynLSTM-for-NER?tab=readme-ov-file#related-repo), NAACL 2021
["Edge-Enhanced Graph Convolution Networks for Event Detection with Syntactic Relation"](https://github.com/cuishiyao96/eegcned), EMNLP 2020
["Attention Guided Graph Convolutional Networks for Relation Extraction"](https://github.com/Cartus/AGGCN_TACRED), ACL 2019
