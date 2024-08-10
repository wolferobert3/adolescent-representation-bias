# Adolescent Representational Bias in AI Repository
This is the code repository for the AIES'24 paper ***Representation Bias of Adolescents in AI: A Bilingual, Bicultural Study***, available at [https://arxiv.org/pdf/2408.01961](https://arxiv.org/pdf/2408.01961).

### 1. Structure

This project required three primary workflows: one to obtain data from AI models, a second to analyze that data, and a third to compare it to human data. The first workflow utilizes the files prefixed with "textgen_", which prompt generative language models for output, and the files in the "training_notebooks" directory, which train a Nepali embedding. The results of this workflow are saved into the "textgen_results" directory; we've included the model outputs for reference.

The analysis workflow uses the "word_embedding_analysis" notebook and the "word_embedding_clustering" notebooks to compute associations and cluster them as seen in the paper. The results are saved in the "swe_results" and "swe_results_clustering" directories, respectively, which include our outputs in this repository.

Finally, the human-AI comparison workflow uses the "human_ai_correlations" and "human_word_clustering" notebooks, operating on data obtained from our participants, which is contained in the "human_data" directory. The results of these analyses are saved in the "human_ai_comparison_results" and "human_results_clustering" directories, which include our outputs in this repository.

### 2. Requirements

The requirements file includes the libraries needed to run the analyses. We recommend creating a unique environment for running the project (for example, with conda, `conda create -n "representationbias" python=3.11`) and then installing the requirements (`pip install -r requirements.txt`). Note that some of the notebooks are intended to be run on Google Colab, especially those using a GPU to run a 4-bit quantized LLaMA-2 model; we've left in the code to do that. 

The scripts to train Nepali word embeddings found in the training_notebooks subdirectory downloads the GloVe code available at [https://github.com/stanfordnlp/GloVe](https://github.com/stanfordnlp/GloVe). Training data for the embeddings can be found on the NepBERTa authors' website at [https://nepberta.github.io/](https://nepberta.github.io/) by clicking on their "Training Data" link. Please make sure to cite the authors of GloVe and NepBERTa if using the code/data in those repositories.

### 3. Token Access

A HuggingFace account is needed to use some of the HuggingFace Hub functionality, which includes verifying access to models like LLaMA-2. You can get a token from this page, after logging into your HuggingFace account: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 4. Paper & Citation

Below follows the information to cite our paper:

```bibtex
@article{wolfe2024representation,
  title={Representation Bias of Adolescents in AI: A Bilingual, Bicultural Study},
  author={Wolfe, Robert and Dangol, Aayushi and Howe, Bill and Hiniker, Alexis},
  journal={arXiv preprint arXiv:2408.01961},
  year={2024}
}
```

### 5. Other Resources

This repository is primarily a reference implementation intended for reproducibility. However, this also means that our repo might not be the best starting place for everyone who needs to learn representations or use the most recent models for the purpose of studying bias in AI. There are many great resources for using the technologies employed in the paper. Some resources we found helpful include:

- [https://fasttext.cc/docs/en/unsupervised-tutorial.html](**https://fasttext.cc/docs/en/unsupervised-tutorial.html**): A tutorial on training unsupervised fasttext word embeddings using the command line of via a python library.
- [https://nlp.stanford.edu/projects/glove/](**https://nlp.stanford.edu/projects/glove/**): An introduction to the GloVe embeddings with useful information about training a model on a custom corpus, and using it to compare word similarities.
- [https://github.com/artidoro/qlora](**https://github.com/artidoro/qlora**): The official repository for the qLoRA technology that enables quantized language models to be fine-tuned with insertable weight matrices. Includes helpful colab demos.
- [https://huggingface.co/learn/nlp-course/chapter1/1](**https://huggingface.co/learn/nlp-course/chapter1/1**): The HuggingFace NLP course, a great way to get quickly caught up on the state of the art in open language technologies.