# Surveillance Object Detection Project

A comprehensive computer vision project for object detection in surveillance scenarios using multiple state-of-the-art models including HRFuser, SAF-FCOS, and MT-DETR.

## 🚀 Features

- **Out-of-the-box detection**: Run Faster R-CNN on images and videos with a single command
- **Multiple Model Support**: Integration with HRFuser, SAF-FCOS, and MT-DETR object detection models
- **Data Processing**: Tools for converting between KITTI, COCO, and custom annotation formats
- **Visualization**: Comprehensive plotting and analysis tools for model performance metrics
- **Video Processing**: Create videos from image sequences with customizable FPS
- **Dataset Management**: Utilities for dataset conversion, resizing, and annotation handling

## 📋 Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Surveillance
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## ⚡ Quick start — run detection

Detection uses **Faster R-CNN** (torchvision) with **COCO** classes. GPU is used automatically when available.

**Single image:**

```bash
python scripts/detect_image.py --input 0855.jpg --output 0855_det.jpg
```

**Video** (default: first **60 frames**; progress bar + FPS at the end):

```bash
python scripts/detect_video.py --input abc.mp4 --output abc_det.mp4
python scripts/detect_video.py -i abc.mp4 -m 0   # full video (no 60-frame limit)
```

**Unified CLI** (image / video / live in one entry point):

```bash
python scripts/surveillance_cli.py image -i 0855.jpg -o 0855_det.jpg
python scripts/surveillance_cli.py video -i abc.mp4 -o abc_det.mp4
python scripts/surveillance_cli.py live --source 0
```

**Live camera (real-time):**

```bash
# Webcam (camera 0); press 'q' to quit
python scripts/surveillance_cli.py live --source 0

# Or use the helper script (activates .venv if present)
chmod +x scripts/run_live_camera.sh
./scripts/run_live_camera.sh

# Save live feed to file
./scripts/run_live_camera.sh --output live_out.mp4
```

**RTSP stream:** `python scripts/surveillance_cli.py live --source "rtsp://..."`

## 📁 Project Structure

```
Surveillance/
├── scripts/                    # Main Python scripts
│   ├── detection_common.py           # Shared model, batch inference, drawing, COCO export
│   ├── detection_runner.py           # Config, validation, image/video/live runners
│   ├── surveillance_cli.py         # Unified CLI: image | video | live
│   ├── detect_image.py               # Object detection on a single image
│   ├── detect_video.py               # Object detection on video
│   ├── config_loader.py               # YAML config loader
│   ├── create_video_from_images.py   # Video creation from image sequences
│   ├── generate_sample_images.py     # Generate synthetic test images
│   ├── convert_pkl_to_json.py        # Convert pickle to JSON format
│   ├── kitti_to_coco/                # KITTI to COCO conversion tools
│   ├── result_plots/                 # Model performance visualization
│   └── ...
├── config/
│   └── config.example.yaml           # Example config (copy to config.yaml)
├── tests/                      # Pytest tests
│   ├── test_detection_common.py      # Unit tests for detection_common
│   └── test_detection_integration.py # Integration test (run detection on sample image)
├── bash_scripts/               # Training and experiment scripts
├── sample_images/               # Sample test images
└── requirements.txt            # Python dependencies
```

## 🎯 Usage

### Object detection (image, video, live)

| Command | Description |
|--------|-------------|
| `scripts/detect_image.py -i IN -o OUT` | Detect on one image, save visualized image |
| `scripts/detect_video.py -i IN -o OUT` | Detect on video, save annotated video + report FPS |
| `scripts/surveillance_cli.py image/video/live ...` | Same options via unified CLI |

**Common options (image & video):**

- `-t, --score-threshold` — Confidence threshold (default: 0.5 or from config)
- `-c, --classes` — Only keep these COCO classes, e.g. `person,car`
- `--model -M` — Model: `fasterrcnn_resnet50_fpn_v2` (default, better accuracy), `retinanet_resnet50_fpn_v2`, `fcos_resnet50_fpn`, etc.
- `--tta` — Test-time augmentation (multi-scale + flip) for higher accuracy (slower)
- `--robust` — Weather-robust preprocessing (CLAHE + denoise) for low light, fog, rain
- `--checkpoint PATH` — Use a fine-tuned checkpoint (from `train_detector.py`)
- `--save-json PATH` — Save COCO-style detections to JSON
- `--fp16` — Use half precision on GPU (faster, less memory)
- `--device auto|cuda|cpu` — Override device

**Video-only:**

- `-b, --batch-size N` — Frames per batch (default: 1; increase for faster GPU use)
- `-n, --every-n-frames N` — Run detection every N frames (1 = every frame)
- `-m, --max-frames N` — Process at most N frames (default: 60). Use `-m 0` for full video.

**Live:**

- `--source 0` or `rtsp://...` — Camera index or RTSP URL
- `--output PATH` — Optional: save output video
- `--no-display` — Don’t show window (use with `--output`)

Detection uses `detection_common.py` (Faster R-CNN / RetinaNet / FCOS, COCO 80 classes), class-colored boxes, and optional config from `config/config.yaml` (see **Configuration**).

### Improving accuracy and weather robustness

- **Stronger model (default):** The default is `fasterrcnn_resnet50_fpn_v2` for better accuracy. Use `--model retinanet_resnet50_fpn_v2` or `--model fcos_resnet50_fpn` to try other backbones.
- **Weather / low light:** Use `--robust` to enable CLAHE + denoising so detection works better in fog, rain, and low light.
- **Maximum accuracy:** Use `--tta` for test-time augmentation (multi-scale + flip, then NMS). Slower but often improves recall.
- **Fine-tune on your data:** Use the training script with heavy weather-style augmentation so the model sees many lighting/weather variants:

```bash
# Fine-tune on COCO-format data (images + instances_*.json) with augmentation
python scripts/train_detector.py --data-dir /path/to/coco/root --annotations annotations/instances_train2017.json --epochs 5 --output weights/finetuned.pt

# Then run detection with your checkpoint
python scripts/detect_image.py -i 0855.jpg -o out.jpg --checkpoint weights/finetuned.pt
```

See `scripts/train_detector.py` for options (`--model`, `--epochs`, `--lr`, etc.). The script applies brightness/contrast, blur, noise, and color jitter to simulate different weather and lighting.

### Create Video from Images

```bash
python scripts/create_video_from_images.py
```

Follow the prompts to specify:
- Image path or glob pattern (e.g., `sample_images/*.png`)
- Output video path (default: `output_video.mp4`)
- FPS (default: 6)

### Generate Sample Images

```bash
python scripts/generate_sample_images.py
```

This generates synthetic test images in the `sample_images/` directory.

### Convert Pickle to JSON

```bash
python scripts/convert_pkl_to_json.py
```

Edit the script to specify your input/output paths.

### Model Performance Visualization

Navigate to `scripts/result_plots/` and run the appropriate plotting scripts for your model:
- HRFuser: `scripts/result_plots/hrfuser/`
- SAF-FCOS: `scripts/result_plots/saf_fcos_vs_hrfuser/`
- MT-DETR: `scripts/result_plots/mt_detr/`

## 🔧 Configuration

Copy `config/config.example.yaml` to `config/config.yaml` (or project root `config.yaml`) and adjust. All keys are optional; detection scripts fall back to CLI defaults.

**Detection section** (used by `detect_image`, `detect_video`, and `surveillance_cli`):

| Key | Description | Default |
|-----|-------------|--------|
| `detection.score_threshold` | Min confidence to keep detections | `0.5` |
| `detection.device` | `"cuda"`, `"cpu"`, or `"auto"` | `"cuda"` |
| `detection.use_fp16` | Half precision on GPU | `false` |
| `detection.batch_size` | Video: frames per batch | `4` |
| `detection.every_n_frames` | Video: run detection every N frames | `1` |
| `detection.class_filter` | Only these COCO classes, e.g. `["person", "car"]` | `[]` (all) |
| `detection.model_type` | Model (e.g. `fasterrcnn_resnet50_fpn_v2`, `retinanet_resnet50_fpn_v2`) | `fasterrcnn_resnet50_fpn_v2` |
| `detection.use_tta` | Use test-time augmentation for higher accuracy | `false` |
| `detection.robust_preprocess` | Weather-robust preprocessing (low light/fog/rain) | `false` |
| `detection.checkpoint_path` | Path to fine-tuned checkpoint | `null` |

Other sections: `paths`, `video`, `image`, `models`, `datasets`, `processing` — see `config/config.example.yaml` for comments.

## 📊 Detection classes (COCO)

The built-in detector uses **COCO** classes. Common ones for surveillance:

- **person**, **bicycle**, **car**, **motorcycle**, **bus**, **truck**
- **traffic light**, **stop sign**, **parking meter**
- **backpack**, **handbag**, **cell phone**

Use `--classes person,car` to restrict to specific classes. Full list: see `COCO_INSTANCE_CATEGORY_NAMES` in `scripts/detection_common.py`.

## 📊 Models

### HRFuser
High-Resolution Feature Fusion model for multi-modal object detection.

### SAF-FCOS
Scale-Aware Feature COS (FCOS) model optimized for surveillance scenarios.

### MT-DETR
Multi-Task DETR (Detection Transformer) model with various fusion strategies.

## 🧪 Tests

Run detection unit and integration tests:

```bash
pytest tests/ -v
```

- `tests/test_detection_common.py` — COCO names, class filter, draw, load model, inference
- `tests/test_detection_integration.py` — Runs detection on `sample_images/frame_0000.png` and checks output image + JSON (skipped if sample missing)

## 🧪 Experiment Tracking

Experiment logs and results are stored in `bash_scripts/` subdirectories:
- Training logs
- Model checkpoints
- Performance metrics
- Configuration files

## 📝 Notes

- Ensure you have sufficient disk space for datasets and model weights
- Some scripts contain hardcoded paths that may need adjustment for your environment
- GPU is recommended for model training and inference

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- HRFuser: [Add citation]
- SAF-FCOS: [Add citation]
- MT-DETR: [Add citation]
