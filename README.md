# Video Motion Trajectory Visualizer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

This is a simple Gradio-based GUI application for extracting and visualizing motion trajectories from videos.

## Example Result

![Example Trajectory](./assets/trajectory_result.jpg){width=50%}
*(Example showing a robotic arm's motion trajectory fading smoothly over time)*

## Prerequisites

- **Python**: 3.9 or higher (3.10+ recommended)
- **System Requirements**: 
  - `ffmpeg` must be installed and accessible in your system's PATH. The tool relies on FFmpeg for automatic video transcoding to ensure web browser playback compatibility.
    - Ubuntu/Debian: `sudo apt-get install ffmpeg`
    - macOS: `brew install ffmpeg`
    - Windows: Install via `winget install ffmpeg` or download from the official site.

## Installation / Environment Setup

We highly recommend using `conda` to set up an isolated virtual environment to avoid dependency conflicts:

```bash
# 1. Create a new conda environment (Python 3.10 is thoroughly tested)
conda create -n traj_vis python=3.10 -y

# 2. Activate the environment
conda activate traj_vis

# 3. Install the required Python packages
pip install -r requirements.txt
```

*Note: Gradio updates frequently. This code functions perfectly with recent Gradio 4.x / 5.x / 6.x versions. If you encounter any UI component issues, consider freezing Gradio to a stable version like `pip install gradio==4.44.1`.*

## Execution

Ensure your environment is activated, then simply run the main GUI application script:

```bash
python gui_app.py
```

The terminal will log the local server address (typically `http://127.0.0.1:7860`). Open this URL in any modern web browser to access the tool.

### Basic Workflow:
1. **Upload Video**: Drag & drop an MP4 (or other formats). The system triggers automatic web-safe H.264+AAC conversion.
2. **Trim & Configure**: Use the timeline sliders to select exact start/end durations and set the sampling `N` frames.
3. **Render Trajectory**: Choose an extraction mode (e.g., `focus_endpoints`) and paint color, then click render. 
4. **Interactive Brush**: Post-generation, select specific layers (like the Background or Original First Frame) from the dropdown and use the feathered brush on the canvas to seamlessly reveal or hide visual elements.
5. **Export**: Confirm your edits to merge layers and download the final high-resolution JPEG and PDF files.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
