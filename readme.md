<p align="center">
    <img src="assets/d2d_logo.png" width="550"></a>
    <br />
    <br />
    <a href="https://github.com/pytorch/fairseq/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
</p>

- [D2D Toolkits: A fast and versatile document-level machine translation toolkits](#d2d-toolkits-a-fast-and-versatile-document-level-machine-translation-toolkits)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Prepare dataset](#prepare-dataset)
    - [Training](#training)
    - [Generation](#generation)
    - [Length Bias DNMT](#length-bias-dnmt)
  - [License](#license)
  - [Citation](#citation)


# D2D Toolkits: A fast and versatile document-level machine translation toolkits
This repository contains the source code for **D2D Toolkits**, A fast and versatile document-level machine translation toolkits for [fairseq](https://github.com/facebookresearch/fairseq).

We also provide our latest document-level machine translation methods:
- [*Addressing the Length Bias Problem in Document-Level Neural Machine Translation*](https://arxiv.org/abs/2311.11601)

## Features
- Fast and efficient document-level data indexing and saving.
- Reliable decoding strategies for document-level machine translation.
- Support various document-level machine translation methods:
  - Document to document translation
  - Document to sentence translation
  - Contextualized translation
- Support various document level data augmentation methods:
  - [*Dynamic Length Sampling*](https://arxiv.org/abs/2311.11601)
  - [Divide and Rule](https://arxiv.org/abs/2103.17151)
  - [Multi Resolution](https://arxiv.org/abs/2010.08961)

## Installation
First install the latest [fairseq](https://github.com/facebookresearch/fairseq).

```bash
git clone https://github.com/facebookresearch/fairseq
cd fairseq
pip install ./
```

Then install the required packages for D2D Toolkits.

```bash
git clone https://github.com/salvation-z/LengthBiasDNMT
cd LengthBiasDNMT
pip install -r requirements.txt
```

## Usage
### Prepare dataset
We provide plenty of tools for preprocessing document-level datasets. Here is an example for preparing [Europarl10](https://www.statmt.org/europarl) dataset.

```bash
bash scripts/prepare_europarl10.sh
```

This scripts will automaticly download the Europarl10 dataset and **prepare the document index** for training. For more details about the document index, please refer to the scripts and the source code.

### Training
D2D Toolkits can be used as a standard plugin for fairseq. Add following arguments to fairseq-train for using D2D Tookits.
```
--user-dir D2D_PATH
                    enable d2d plugin for fairseq
--task document_translation 
                    enable d2d plugin for fairseq
--data-type DATA_TYPE
                    specify how texts are data type, e.g. seg2seg, sent2sent, doc2sent, context, hybrid, divide
--allow-mixup       enable document concatenation
--use-tags          add document tags to the input
--use-mask          add local/global masks to the input
```

### Generation
To enable document level decoding, add following arguments to fairseq-generate for your generation.
```
--user-dir D2D_PATH
                    enable d2d plugin for fairseq
--task document_translation
                    enable d2d plugin for fairseq
--data-type DATA_TYPE
                    specify how texts are data type, e.g. seg2seg, sent2sent, doc2sent
--context-window CONTEXT_WINDOW
                    specify the context window size for Slide Decoding
--slide-decode      enable Slide Decoding
--use-tags          add document tags to the input
--use-mask          add local/global masks to the input
--force-decode      force to generate target document that has the same number of sentence as the source document
--allow-longer      do not raise error when the test sequence is longer than the training sequence
```

### Length Bias DNMT
We provide scripts to reproduce our proposed methods in [*Addressing the Length Bias Problem in Document-Level Neural Machine Translation*](https://arxiv.org/abs/2311.11601), please following our [scripts](./scripts/train_length_bias_transformer.sh).

## License
D2D Toolkit is MIT-licensed.


## Citation
```bibtex
@misc{zhang2023addressing,
      title={Addressing the Length Bias Problem in Document-Level Neural Machine Translation}, 
      author={Zhuocheng Zhang and Shuhao Gu and Min Zhang and Yang Feng},
      year={2023},
      eprint={2311.11601},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
```
