# Spatial Deconfounder: Interference-Aware Deconfounding for Spatial Causal Inference 

The Spatial Deconfounder uses a conditional variational autoencoder (CVAE) with a spatial Laplacian prior to recover a smooth substitute confounder from local treatment patterns, addressing both interference and unobserved confounding.

![Deconfounder Architecture](images/deconfounder_architecture.jpg)

## Requirements

To download the necessary requirements:
```
conda env create -f environment.yml
conda activate spacedata
```

## Synthetic Data Generation with Interference

It only supports one GPU for training right now. To generate the data with interference:
```
cd space-data
snakemake -s Snakefile -j --configfile conf/pipeline.yaml
```
To generate the data with a single-cause confounder, run:
```
cd space-data
snakemake -s Snakefile_singlecause -j --configfile conf/pipeline.yaml
```

## Run Benchmarks

Change files in `benchmarks/conf/` to change resources, algorithms, datasets, hyperparams, etc.. The outputs are `jsonl` fles. To run the `pipeline.yaml` file, run:

```
PYTHONPATH=. snakemake -s Snakefile --configfile benchmarks/conf/pipeline.yaml --use-conda --cores all
```

## Run Hyperparameter Tuning

These are used to create the hyperparameter visualization plots. Change files in `benchmarks/conf/` to change resources, algorithms, datasets, hyperparams, etc.. The outputs are `parquet` fles. To run the `pipeline_tune.yaml` file, run:

```
PYTHONPATH=. snakemake -s Snakefile_tune --configfile benchmarks/conf/pipeline_tune.yaml --use-conda --cores all
```
