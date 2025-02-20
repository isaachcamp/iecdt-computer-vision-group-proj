
## Clone repo to directory on JASMIN

```bash
git clone <repo-name>
```

## Install dependencies

Install [uv](https://docs.astral.sh/uv/) to handle virtual environment:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

To install the virtual environment for this project, first navigate to your project repository, then run the following commands:

```bash
uv venv --seed
source .venv/bin/activate
uv sync
```

You also need an extra package `omnicalib`.

```bash
uv pip install https://github.com/tasptz/py-omnicalib/releases/download/1.0.1/omnicalib-1.0.1-py3-none-any.whl
```

This should create a basic working environment.