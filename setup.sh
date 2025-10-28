conda create -n admet python=3.11
conda activate admet

# we will handle most dependencies via pip, because conda has some problems with versions
pip install chemprop gin-config ipython "ray[tune]" seaborn xgboost umap-learn