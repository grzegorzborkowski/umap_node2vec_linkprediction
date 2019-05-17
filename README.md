This code is mainly based on https://github.com/lucashu1/link-prediction

```
@misc{lucas_hu_2018_1408472,
   author       = {Lucas Hu and
                   Thomas Kipf and
                   Gökçen Eraslan},
   title        = {{lucashu1/link-prediction: v0.1: FB and Twitter 
                    Networks}},
   month        = sep,
   year         = 2018,
   doi          = {10.5281/zenodo.1408472},
   url          = {https://doi.org/10.5281/zenodo.1408472}
}
```

To install (you need conda)
```
conda create --name $ENV_NAME python=3.7
source activate $ENV_NAME
conda install --file requirements.txt
pip install lime
pip install umap-learn
```

To run:
```
bash run_experiments.sh
```