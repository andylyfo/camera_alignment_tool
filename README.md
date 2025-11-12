# Camera Alignment Tool

Interactive tool for comparing and aligning cameras using railheads.

![Example](example.png)

## Requirements

```bash
python -m venv venv
pip install -r requirements.txt
```

## Usage

```bash
python camera_alignment_tool.py camera1.png camera2.png
```

**Controls:**
- **Click** - Draw rail lines (2 clicks per line, 2 lines per camera)
- **U** - Undo last action (on selected/last-clicked image)
- **R** - Reset all annotations
- **P** - Polygon Mode (WIP)
- **Q** - Quit
