Choosing your "processing plan"
===============================

#### Introductory notes
ADMET-XSpec lets you switch between splitting and preprocessing data and training models on compositions of that data
quickly, within a single config file. In addition, you can opt in or out of generating visualizations for both the
original body of data and your compositions (mixed splits).

To support this functionality, we use what we call "processing plans". The name reflects the fact that they
control the `ProcessingPipeline`.

#### Goals
After reading this section, you should understand:
1. What processing plans represent and how they connect to the `run` method of ProcessingPipeline
2. Which processing plans are possible and how to configure them
3. The create/load distinction with train/test splits

At a high level, a processing plan controls these 9 steps in the execution of the `ProcessingPipeline`:

```
# Step 1: Load datasets
# Step 2: Visualize raw datasets
# Step 3: Create(*) train/test splits
# Step 4: Save train/test splits
# Step 5: Visualize train/test splits
# Step 6: Load optimized hyperparameters
# Step 7: Optimize hyperparameters
# Step 8: Train and evaluate the model
# Step 9: Refit final model on full dataset
```
These steps are comments taken from the `run` method itself. They are placed between control flow blocks to disable
certain parts from running with an `if-else`.

If you want deeper insight by looking into the methods called by `ProcessingPipeline`, you can check out the `run`
method. However, you will do fine by sticking to one of the processing plans we have provided
in `configs/processing_plans`. These are used in our set of `configs/examples`.

Let's look at `configs/processing_plans/train_optimize.gin`, the one you are likely to use most often.
With this processing plan, the `ProcessingPipeline` will:

1. Load your raw ChEMBL datasets and preprocess them.
2. **Not** visualize the preprocessed datasets, since you set that step to 'False'.
3. Create(\*) your training and test splits, and save them to cache.
4. **Not** visualize the train/test splits, since you set that step to 'False'.
5. **Not** load hyperparameters found to be optimal in a different run, since you set that to 'False'. More on this in Guide 3.4: "Training and optimization".
6. Find optimal hyperparameters for training and train a model on them, as well as refit the model on the entire train+test dataset and generate metrics based on that.

```bash
ProcessingPipeline.do_load_datasets = True  ProcessingPipeline.do_visualize_datasets = False
ProcessingPipeline.do_load_train_test = True
ProcessingPipeline.do_dump_train_test = True
ProcessingPipeline.do_visualize_train_test = False
ProcessingPipeline.do_load_optimized_hyperparams = False
ProcessingPipeline.do_optimize_hyperparams = True
ProcessingPipeline.do_train_model = True
ProcessingPipeline.do_refit_final_model = True
```

The file `configs/processing_plans/_possible_plans.gin` serves as a reminder of what plans you can create whenever
you find yourself outside of our docs.

The (\*) symbol highlights the ambiguity that may arise from referring to the train/test split stage as "creation"
on the one hand and "loading" on the other.

Consider the following two processing plans, which we will call "Just visualize raw" and "Train from select splits":

"Just visualize raw"
```bash
ProcessingPipeline.do_load_datasets = True  ProcessingPipeline.do_visualize_datasets = True
ProcessingPipeline.do_load_train_test = False
ProcessingPipeline.do_dump_train_test = False
ProcessingPipeline.do_visualize_train_test = False
ProcessingPipeline.do_load_optimized_hyperparams = False
ProcessingPipeline.do_optimize_hyperparams = False
ProcessingPipeline.do_train_model = False
ProcessingPipeline.do_refit_final_model = False
```

"Train on select splits"
```bash
ProcessingPipeline.do_load_datasets = False  ProcessingPipeline.do_visualize_datasets = False
ProcessingPipeline.do_load_train_test = True
ProcessingPipeline.do_dump_train_test = False
ProcessingPipeline.do_visualize_train_test = False
ProcessingPipeline.do_load_optimized_hyperparams = False
ProcessingPipeline.do_optimize_hyperparams = True
ProcessingPipeline.do_train_model = True
ProcessingPipeline.do_refit_final_model = True
```

You can see how in the first situation we wish to simply process the original data in some way without training a
model, and in the second situation we do not want to interact with the original data at all, since we already have
generated splits and aim to train a model on those splits. These splits have been saved to disk (in `data/cache`)
and are therefore "loaded".

In this way, ambiguity arises when we run `ProcessingPipeline` on original data, "creating splits" in the process
and immediately proceeding to train a model on those splits. This is, in fact, the only way to create splits: the
original data must be loaded, and then splits must be dumped. That is when they are "created". This is not reflected
in the step's name: `do_load_train_test`. To reconcile this, we can think of the `ProcessingPipeline` as immediately
"loading" the splits it had created.
