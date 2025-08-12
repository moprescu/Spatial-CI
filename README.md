# Spatial-CI
GitHub repository for spatial causal inference research. 


## Space repo
(without space repo and no editing spacebench package)
```
conda env create -f conda.yaml
conda activate benchmarks
pip install "spacebench[all]" --no-deps

```


## Space-data repo
space-data repo (creating data)
```
conda env create -f space-data/requirements.yaml
conda activate spacedata
\
```

## Run code

Look at benchmarks/conf/config.yaml for changing resources (cpu/gpu) and num_samples
Look at benchmarks/conf/pipeline.yaml to change which algos/datasets
Look at benchmarks/conf/algo/yaml files because you may want to change range of values considered for hyperparams
Outputs are .jsonl files
```
PYTHONPATH=. snakemake --configfile benchmarks/conf/pipeline.yaml -C concurrency=1 cpus_per_task=1 --use-conda -j=10
```

## Debugging

To conda environment add to jupyter kernel:
```
conda activate benchmarks
python -m ipykernel install --user --name benchmarks --display-name "Python (benchmarks)"
```

