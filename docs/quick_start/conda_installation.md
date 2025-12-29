Installation with conda
=======================

```shell
conda create -n admet python=3.11.8
conda activate admet_xspec
conda install rdkit seaborn conda-forge::py-xgboost conda-forge::ray-all

pip install -r requirements.txt

# Dev dependencies
pre-commit install
```

#### Run example with conda
```shell
# if you haven't already:
conda activate admet_xspec

python -m process --cfg configs/examples/train_optimize_rf_clf.gin
```

##### Optional, recommended: integrate conda with PyCharm

In your Status Bar (bottom right corner of IDE), click on the button which gives a "Current Interpreter" hover.
Select "Add New Interpreter" -> "Add Local Interpreter" -> "Select Existing".

Change the type to 'conda'. The conda path should be that of your system conda installation, the Environment path
should be that of the conda environment you just created. If it does not appear, attempt to 'Reload environments'.
