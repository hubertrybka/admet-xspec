Installation with uv
====================

Follow the "Install uv" step at the [official docs](https://docs.astral.sh/uv/#__tabbed_1_1) to set up uv.
Then, have uv register a .venv within the current directory and install packages from the lockfile:

```shell
uv init .
uv sync
```

#### Run example with uv

```shell
uv run process.py --cfg configs/examples/train_optimize_rf_clf.gin
```

##### Optional, recommended: integrate uv with PyCharm

In your Status Bar (bottom right corner of IDE), click on the button which gives a "Current Interpreter" hover. Select
"Add New Interpreter" -> "Add Local Interpreter" -> "Select Existing".

Change the type to 'uv'. The uv path should be that of your system uv installation, the Environment path should be
that of the `.venv` that uv created inside of `./admet_prediction`.
