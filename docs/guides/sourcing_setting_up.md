Sourcing and setting up data
============================

#### Introductory note
ADMET-XSpec currently only supports ChEMBL datasets. If you want to add a dataset from a different source, ensure it is preprocessed to the same format as these examples:
```bash
./data/datasets/permeability/B3DB/binary_classification or
./data/datasets/permeability/PAMPA/binary_classification
```

Before we start: recall from 1.2: "Killer features" that directory structure is there to help you organize your data. When using our tool, **`friendly_name` is the name of the game**. You will use `friendly_names` in your experiment `.gin` configs to specify which raw datasets or splits you wish to use for training.

### Setting up
#### Goals
After reading this section, you should understand:
1. Where to place raw datasets downloaded from ChEMBL
2. How **params.yaml** defines how `ProcessingPipeline` will handle your dataset
3. How to use regression datasets for binary classification

Suppose you have a human and mouse dataset for IC50 regression in acetylcholinesterase (AChE). We will discuss how to set those up to train both `Regressors` and `BinaryClassifiers` within ADMET-XSpec.

Let's replay our thought process when setting up our AChE IC50 data so you can follow along.

We placed our human data in `/data/datasets/AChE/human/regression`. Since we want the same data source converted to the binary classification setting, we set up a copy in `/data/datasets/AChE/human/binary_classification`. This copying of data to convert from regression to classification is the only instance where we do this in ADMET-XSpec. The rest is handled programmatically.

Our datasets are now:
`/data/datasets/AChE/human/regression/DOWNLOAD-(long_name_1).csv`
`/data/datasets/AChE/human/binary_classification/DOWNLOAD-(long_name_1).csv` _long name replaces the seriously long string in the ChEMBL dataset name_.

We also set up the mouse dataset in similar fashion: `/data/datasets/AChE/mouse/regression/DOWNLOAD-(long_name_2).csv` `/data/datasets/AChE/mouse/binary_classification/DOWNLOAD-(long_name_2).csv`

**Here we discuss the contents of `params.yaml` for each of these datasets**, with the exception of `mouse/regression`, as it will not be particularly informative:

`AChE/human/regression/params.yaml`:
```yaml
1.  friendly_name: "AChE_human_IC50"
2.  raw_or_derived: "raw"
3.  category: "AChE_IC50"
4.  is_chembl: true
5.  task_setting: "regression"
6.  filter_criteria:
7.      Standard Units:
8.        - "nM"
9.      Standard Relation:
10.       - "'='"
11.     Standard Type:
12.       - "IC50"
13. label_transformations:
14.   - "log10"
15.   - "negate"
```

Line 1 provides the dataset's `friendly_name`, which you will use to refer to it in your experiment's `.gin` configs.

Line 2 states that the `.csv` accompanying this `.yaml` (the downloaded dataset) is data in its `raw`, original form, in contrast to splits, which are outputted by the `ProcessingPipeline` when integrating data and are always labelled as `derived` in their accompanying `.yaml` (yes, they get a `friendly_name` too!).

Line 3 is no longer used and is slated for removal :)

Line 4 notes whether the dataset was downloaded from ChEMBL. This is added to accommodate datasets from other sources that have already been preprocessed by the user. For now, you can safely assume ADMET-XSpec can only accommodate ChEMBL datasets and proceed accordingly.

Line 5 states how the dataset should be treated: whether it is a `regression` or `binary_classification` dataset. Since we offer the feature of running classification on datasets originally labelled as regression, we wanted to be able to use the same `friendly_names`, with the `ProcessingPipeline` distinguishing between task settings for us. This accomplishes that.

Lines 6-12 outline the type of values a row must have (based on the column values in ChEMBL datasets) to be retained and not excluded from training. That is, for a molecule (row) to remain in the training dataset, it must have `nM` in the `Standard Units` column, `'='` in the `Standard Relation` column, and `IC50` in the `Standard Type` column.

Lines 13-15 provide the label transformations to be used before training.

`AChE/mouse/binary_classification/params.yaml`:
```yaml
1.  friendly_name: "AChE_mouse_IC50"
2.  raw_or_derived: "raw"
3.  category: "AChE_IC50"
4.  is_chembl: true
5.  task_setting: "binary_classification"
6.  filter_criteria:
7.      Standard Units:
8.        - "nM"
9.      Standard Type:
10.       - "IC50"
11. threshold: null
12. threshold_source: "AChE_human_IC50"
```

Lines 1-10 follow the same configuration as `AChE/human/regression/params.yaml`.

Lines 11 and 12 are slightly more complex:

This dataset has the same `.csv` as its regression counterpart, but the training data it produces (splits; `derived` data) will have binary classification labels. To convert to binary classification, we need a threshold for our regression values.

Line 12 specifies where to source our threshold. If we navigated to the dataset specified in `threshold_source`, its Line 11 would contain, for instance, `"median"`. This means that to obtain our `0` vs. `1` label threshold, we take the median of regression values from `threshold_source`.

For this reason, Line 11 is `null` here, as we do not intend to use this dataset as the threshold source for any other dataset.

`AChE/human/binary_classification/params.yaml`:
```yaml
1.  friendly_name: "AChE_human_IC50"
2.  raw_or_derived: "raw"
3.  category: "AChE_IC50"
4.  is_chembl: true
5.  task_setting: "binary_classification"
6.  filter_criteria:
7.      Standard Units:
8.        - "nM"
9.      Standard Type:
10.       - "IC50"
11. threshold: "median"
```

Line 11 here clarifies and completes the relation to Lines 11-12 in `mouse/binary_classification`.
