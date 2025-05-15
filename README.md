# Post-training of DNA Foundation Models to Analyze DNA Sequences

## Overview

This repository provides code for post-training and evaluation of DNABERT-2, a DNA foundation model, to study DNA sequenes from different species. This code was used to analyze DNA sequences from different species of Caenorhabditis, a genus of nematodes with relatively small genomes.

## Data

Selected reference genomes were retrieved from the NIH National Library of Medicine database. The files were processed to obtain a desired length per line with 50% overlap between sequences. Chromosome I of C. elegans, a well-studied model organism, was set aside for testing and all other genomes were used for training and validation.

## Model

As mentioned DNABERT-2 model was used in this work. This model can be installed using the following code:
```bash 
  git clone https://github.com/Zhihan1996/DNABERT_2.git 
  cd DNABERT_2
```
