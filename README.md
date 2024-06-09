# Graphical Models Tutorial, Using GitHub Codespaces

## Setup

1) Fork this repository
2) Click on the green 'code' button
3) Navigate to 'Codespaces'
4) Click the '+' button to create a codespace based on your forked repository
5) Run all cells in `notebooks/setup.ipynb`

The last step will add the relevant packages and data to your codespace.  The actual tutorial is in the file `notebooks/experiment.ipynb`, but do not start going through that until told.

## Blurb

In this repository, we will investigate the following dataset:

* [Spatio-temporal immune zonation of the human kidney](https://www.ebi.ac.uk/gxa/sc/experiments/E-HCAD-10/results/tsne), with [an associated paper](https://pubmed.ncbi.nlm.nih.gov/31604275/).

We will investigate it with two graphical models:

1) GLasso
    * Makes independence assumption on the samples
    * Uses regularization for more robust inference
2) GmGM
    * Does not make an independence assumption; finds graphs for both genes and cells
    * Uses thresholding rather than regularization