
## Install dependencies

Install [uv](https://docs.astral.sh/uv/) to handle virtual environment:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

To install the virtual environment for this project, first navigate to the project repository:

```bash
cd /gws/nopw/j04/iecdt/JERMIT_the_Frog/iecdt-computer-vision-group-proj
```

Then run the following commands:

```bash
uv venv --seed
source .venv/bin/activate
uv sync
```

This should create a basic working environment.