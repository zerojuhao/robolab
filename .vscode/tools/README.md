# VSCode Tools

This directory contains tools for setting up VSCode for RoboLab development.

## setup_vscode.py

This script configures VSCode to recognize IsaacLab and rsl_rl paths for Python IntelliSense and code navigation.

### Usage

Run the script from the repository root:

```bash
python3 .vscode/tools/setup_vscode.py
```

This will create or update `.vscode/settings.json` with the following paths added to `python.analysis.extraPaths`:

- `${workspaceFolder}/../IsaacLab/source/isaaclab_rl`
- `${workspaceFolder}/../IsaacLab/source/isaaclab_tasks`
- `${workspaceFolder}/../IsaacLab/source/isaaclab_assets`
- `${workspaceFolder}/../IsaacLab/source/isaaclab_mimic`
- `${workspaceFolder}/../IsaacLab/source/isaaclab`
- `${workspaceFolder}/../rsl_rl`

### Prerequisites

The script assumes that IsaacLab and rsl_rl are located in the parent directory alongside the robolab repository:

```
atom_train/
├── IsaacLab/
├── robolab/
└── rsl_rl/
```

### Options

- `--isaaclab-path ISAACLAB_PATH`: Optional path to IsaacLab directory (not currently used, reserved for future functionality)

### Notes

- The script is idempotent - running it multiple times won't duplicate paths
- Existing VSCode settings are preserved
- The generated `settings.json` file is ignored by git (configured in `.gitignore`)
