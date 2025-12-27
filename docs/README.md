### Setting up dependencies for Read the Docs
```bash
touch docs/requirements.in
printf sphinx==9.0.4 > docs/requirements.in
uv pip freeze >> docs/requirements.in
uv run pip-compile docs/requirements.in
```
