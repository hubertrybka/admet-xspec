Killer features
===============
1. Pick and mix featurizers, predictors and similarity filters. Track your experiments in an easily-modifiable file.

We use Google's [gin-config](https://github.com/google/gin-config), a "lightweight configuration framework for Python". The featurizer, predictory and filter components of ADMET-XSpec can be found in the project root directory under `./configs`:
```
├── configs
│   ...
│   ├── featurizers
│   ├── predictors
│   ├── sim_filters
│   ...
```

You can set up your own experiment by following our examples under `./configs/examples`. Simply copy these `.gin` files and modify them, naming them whatever you like to track your experiments.

To understand more about how to configure `ProcessingPipeline` and produce a model, visit Guide 3.3: "Training and choosing your `processing_plan`".


2. Add new data in whichever directory structure you like: provide it with a `friendly_name`, outline a set of basic facts about it, and be on your way.

The datasets we used for our experiments are in `./data/datasets`, with the directory structure serving as an organizational aid. The source of truth about a dataset - including its `friendly_name` for locating it - can be found in its accompanying `params.yaml` file. Here's an example:

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

Twelve lines of configuration isn't bad at all! You can find guidance on configuring `params.yaml` in Guide 3.1: "Sourcing and setting up data".


3. Track splits, models, metrics, and logs. Every product of your `ProcessingPipeline` runs is stored in `data/cache`.

Let this `tree ./data/cache` output serve as an example:
```bash
├── models
│   └── LightGBM_clf_ecfp_featurizer_4b52a
│       └── scaffold_e4737_tanimoto_5p_filter_c2805_91da5
│           ├── hyperparams.yaml
│           ├── metrics.yaml
│           ├── model_final_refit.pkl
│           ├── model_metadata.yaml
│           ├── model.pkl
│           ├── operative_config.gin
│           └── training_log
│               └── console.log
└── splits
    ├── registry.txt
    ├── scaffold_e4737_tanimoto_5p_filter_c2805_660d3
    │   ├── console.log
    │   ├── operative_config.gin
    │   ├── test
    │   │   ├── data.csv
    │   │   └── params.yaml
    │   └── train
    │       ├── data.csv
    │       └── params.yaml
```

There are two subdirectories that interest us: `models` and `splits`.

The first contains trained models ready for use in `InferencePipeline`. The following files are also outputted:
1. Hyperparameters (of particular interest when running optimization; see Guide 3.3: "Training & choosing your processing plan")
2. Metrics on the test set
3. Final model refits on the entire training data
4. Pipeline metadata related to ADMET-XSpec
5. The `.gin` config settings, referred to as an "operative_config"
6. The training log (i.e., the output of `logging.info` or whatever level you set)

The second contains dataset splits ready to be reused and reconfigured for subsequent `ProcessingPipeline` runs. You can run the `ProcessingPipeline` without training a model and use it only for data splitting. You can then feed the split data to train a model of your choice. You can also reuse split data from previous training runs.

Here's an outline of the contents of `splits`:

- `registry.txt`, which contains a list of all splits (their `friendly_names`) within `data/cache/splits`, updated on each `ProcessingPipeline` run Within each run's resulting splits (e.g., `scaffold_e4737_tanimoto_5p_filter_c2805_660d3`):
    1. The splitting log (i.e., the output of `logging`)
    2. The `.gin` config settings
    3. The created train split (_derived dataset_), along with its `params.yaml`
    4. The created test split (_derived dataset_), along with its `params.yaml`
