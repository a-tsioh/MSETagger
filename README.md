# THIS IS A WORK IN PROGRESS

Re-implementation of the system described in [this paper](https://hal.archives-ouvertes.fr/LIMSI/hal-01793092v1), a POS tagger designed for low resource languages.
The goal is to make it available and usable for anyone facing low resources issues in NLP.


# MSETagger
POS tagging for low-resource languages, using specialized MorphoSyntactic Embeddings

# Dependecies

The tagger is built on top of [yaset](https://github.com/jtourille/yaset) for the Bi-LSTM tagger part and [mimick](https://github.com/yuvalpinter/Mimick) to compute embeddings of OOVs

# Requirements

* SBT 
* a way to create virtual environments for Python
* some data for your favorite language

# Installation

* clone the repo
* create two virtual env for python 2 and 3 (sorry, yaset is in python3 and mimick in 2...) and `pip install -r requirements(2|3).txt` for each 
* `sbt compile` the scala code

# Usage

Every options are set in the application.conf file, it includes:

* env section
  * path to the two python interpreters
* tokenizer section
  * strategy is either a `basic` whitespace tokenizer or a call to an `external` command 
  * if `strategy` is `external`, you need to specify a `command` to be called
* embeddings section
  * `file` is the path to/from which embeddings are saved
  * `oovfile` list the forms for which embeddings are to be generated
  * `ndim` is the number of dimensions of the embeddings
  * `ws` is the window size
  * `nboccmin` is a minimum number of occurrences of a form to have the embeddings computed my MSE (otherwise it will be computed as an OOV)
* `yaset.model` is a zip file containing all the data created/needed by yaset
* corpora section defines the paths to the corpus file (to be read and/or written)  
  

