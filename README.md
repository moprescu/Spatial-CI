# Spatial-CI
GitHub repository for spatial causal inference research. 


space repo (rerun benchmarks and update algo)
possibly easier to add an algorithm in space/spacebench/algorithms and rerun it using the benchmarks they have?

## Space repo
two possiblities (with space repo and editing spacebench package):
```
conda env create -f conda.yaml
conda activate benchmarks
cd space
pip install -e ".[all]" --no-deps

```


(without space repo and no editing spacebench package)
```
conda env create -f conda.yaml
conda activate benchmarks
pip install "spacebench[all]" --no-deps

```


## Space-data repo
space-data repo (creating data)
```
conda env create -f requirements.yaml
conda activate spacedata
\
```


## Debugging

To add to jupyter kernel:
```
conda activate benchmarks
python -m ipykernel install --user --name benchmarks --display-name "Python (benchmarks)"
```

