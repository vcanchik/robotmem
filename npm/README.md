# @muscular/robotmem

Thin npm wrapper for the `robotmem` Python package and CLI.

It installs a `robotmem` command that:

- finds a local Python interpreter
- installs `robotmem` into that interpreter on first run when needed
- forwards all CLI arguments to `python -m robotmem`

## Requirements

- Node.js 16+
- Python 3.10+
- `pip` available for the detected Python interpreter

## Install

```bash
npm install -g @muscular/robotmem
```

## Usage

```bash
robotmem
robotmem web --port 6889
```

The first invocation may take longer because the wrapper installs the Python package before starting the MCP server or Web UI.

## Python API

If you want direct library access instead of the wrapper, install the Python package:

```bash
pip install robotmem
```

```python
from robotmem import learn, recall

learn("grip_force=12.5N yields highest grasp success rate")
print(recall("grip force parameters"))
```

## Project

- GitHub: https://github.com/vcanchik/robotmem
- Upstream: https://github.com/robotmem/robotmem
- License: Apache-2.0
