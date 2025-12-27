### Setting up dependencies for Read the Docs
```bash
touch docs/requirements.txt
printf sphinx==9.0.4 > docs/requirements.tx
uv pip freeze >> docs/requirements.txt
```
