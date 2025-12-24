# find all model.pkl files in the current directory and its subdirectories
import pathlib
import yaml
import pandas as pd

def find_model_files(root_dir='data/cache/models'):
    root = pathlib.Path(root_dir)
    model_files = list(root.rglob("model.pkl"))
    return model_files

def get_metrics(model_file):
    path = model_file.parent / "metrics.yaml"
    # read yaml file
    with open(path, 'r') as f:
        metrics = yaml.safe_load(f)
    return metrics

def get_metadata(model_file):
    path = model_file.parent / "model_metadata.yaml"
    # read yaml file
    with open(path, 'r') as f:
        metadata = yaml.safe_load(f)
    return metadata

def get_row_data(model_file):
    metrics = get_metrics(model_file)
    metadata = get_metadata(model_file)
    metadata['Datasets'] = ', '.join(metadata.get('Datasets', []))
    row = {**metrics, **metadata}
    return row

if __name__ == "__main__":
    rows = []
    for file in find_model_files():
        summary = get_row_data(file)
        rows.append(summary)

    df = pd.concat([pd.DataFrame([row]) for row in rows], ignore_index=True, sort=False)
    df.to_csv('model_metrics_summary.csv', index=False)