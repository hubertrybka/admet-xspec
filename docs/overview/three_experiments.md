Set up our repository as outlined in 'Installation'.

1. I want to investigate how integrating mouse and rat data with human data affects predicting the activity or inactivity of a molecule, with respect to IC50 in acetylcholinesterase (AChE), using random forests:

```bash
# run with integrated data
python -m process --cfg configs/examples/train_optimize_rf_clf.gin

# remove integrated data
nano configs/examples/train_optimize_rf_clf.gin

# ...
# 'AChE_mouse_IC50', <- remove this line
# 'AChE_mouse_IC50', <- remove this line
# ...

# run with human-only data
python -m process --cfg configs/examples/train_optimize_rf_clf.gin
```

2. I want to investigate how integrating mouse data with human data affects predicting the value of molecule elimination half-life (t1/2), in the liver, using SVM:

```bash
# run with integrated data
python -m process --cfg configs/examples/train_optimize_svm.gin

# remove integrated data, just like in #1
nano configs/examples/train_optimize_svm.gin

# ...

# run with human-only data
python -m process --cfg configs/examples/train_optimize_svm.gin
```

3. I want to investigate how integrating mouse and rat data with human data affects predicting the value of inhibition (measured in %) of MAO-A using LGBM:

```bash
# run with integrated data
python -m process --cfg configs/examples/train_optimize_lgbm.gin

# remove integrated data, just like in #1 and #2
nano configs/examples/train_optimize_lgbm.gin

# ...

# run with human-only data
python -m process --cfg configs/examples/train_optimize_lgbm.gin
```
