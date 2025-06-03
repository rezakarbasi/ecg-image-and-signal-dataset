# ECG Image and Signal Dataset Generator

This repository provides an open-source Python framework for generating customizable, large-scale synthetic ECG images from signals. Multiple datasets are included to support deep learning-based research on ECG digitization, lead and lead-name detection, and waveform segmentation.

The dataset for waveform segmentation is provided in two versions: normal and overlapping. In the overlapping version, signals from adjacent leads (above or below) are superimposed onto a target lead. At the same time, the corresponding masks remain clean, containing only the true waveform of the target lead.

More information on our paper.
- Paper: should be added after publication.
- Datasets: [https://doi.org/10.5281/zenodo.15484519](https://doi.org/10.5281/zenodo.15484519)

---

## 🧠 Citation
If you use this dataset or code, please cite both the dataset and the paper:

**Paper:**  
*To be added after publication.*

**Dataset:**
> Rahimi, M., Karbasi, R., & Vahabie, A. H. (2025). *An Open-Source Python Framework and Synthetic ECG Image Datasets for Digitization, Lead and Lead Name Detection, and Overlapping Signal Segmentation*. University of Tehran.

Dataset at Zenodo: [https://doi.org/10.5281/zenodo.15484519](https://doi.org/10.5281/zenodo.15484519)

code at GitHub: [https://github.com/rezakarbasi/ecg-image-and-signal-dataset](https://github.com/rezakarbasi/ecg-image-and-signal-dataset)  


---

## 📦 Features

- Generate realistic ECG print-style images in various layouts: `3x1`, `3x4`, `6x2`, `12x1` (in `3x4` and `6x2`, lead II can be printed in full at the bottom)
- Annotate images for:
  - **Lead Region Detection** (YOLO format)
  - **Lead Name Detection** (YOLO format)
  - **Pixel-wise Segmentation** (with both normal and overlapping signals for U-Net)
- Compatible with deep learning models like YOLO and U-Net
- Includes paired image, mask, and time-series data
- Overlapping segmentation support: clean masks for superimposed leads

---

## 📊 Datasets Provided

| Dataset Task               | Format/Annotations                              | Sample Size |
|----------------------------|-------------------------------------------------|-------------|
| ECG Digitization           | ECG images + ground truth signals               | 2,000       |
| Lead & Lead Name Detection | Bounding boxes (YOLO format)                    | 2,000       |
| Segmentation (Normal)      | Cropped leads + masks + ground truth signals    | 20,000      |
| Segmentation (Overlapping) | Overlapping leads + clean masks                 | 102         |

---

## ⚙️ Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Raw Data
Download the raw WFDB files from [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/), unzip them, and place the contents into the `./dataset/signal_dataset/` directory.

### 3. Run the Dataset Generator
```bash
python make_dataset.py \
    --sample_each_lead_train 5 \
    --sample_each_lead_val 2 \
    --sample_each_lead_test 2 \
    --row_height 3
```

Additional arguments available (see `make_dataset.py` for full list):
- `--signal_dataset_path` 
- `--datasets_path`
- `--horizontal_scale`, `--vertical_scale`
- `--show_grid`, `--show_lead_name`, `--fontsize`

### 4. Output Directory
```
.
├── detection-dataset
│   ├── images
│   │   ├── 0_3by1.jpg
│   │   ├── ...
│   └── labels
│       ├── 0_3by1.txt
│       └── ...
├── digitization-dataset
│   ├── 0_3by1
│   │   ├── 0_3by1.jpg
│   │   ├── 0_3by1_lead_0.json
│   │   ├── 0_3by1_lead_1.json
│   │   └── 0_3by1_lead_2.json
│   └── ...
├── overlap-dataset
│   ├── with-overlap
│   │   ├── image
│   │   │   ├── 699_3by4_lead_10.bmp
│   │   │   └── ...
│       ├── mask-bmp
│   │   │   ├── 699_3by4_lead_10.bmp
│   │   │   └── ...
│   │   ├── mask-png
│   │   │   ├── 699_3by4_lead_10.png
│   │   │   └── ...
│   │   └── signal-json
│   │       ├── 699_3by4_lead_10.json
│   │       └── ...
│   └── without-overlap
│       ├── image
│       │   ├── 699_3by4_lead_0.png
│       │   └── ...
│       ├── mask-bmp
│       │   ├── 699_3by4_lead_0.bmp
│       │   └── ...
│       ├── mask-png
│       │   ├── 699_3by4_lead_0.png
│       │   └── ...
│       └── signal-json
│           ├── 699_3by4_lead_0.json
│           └── ...
├── segmentation
│   ├── image
│   │   ├── 0_3by1_lead_0.png
│   │   └── ...
│   ├── mask-bmp
│   │   ├── 0_3by1_lead_0.bmp
│   │   └── ...
│   ├── mask-png
│   │   ├── 0_3by1_lead_0.png
│   │   └── ...
│   └── signal-json
│       ├── 0_3by1_lead_0.json
│       └── ...
└── logs.json

```

---

## 📬 Contact
Feel free to open an issue or discussion if you need help.

For questions, please contact:
- Masoud Rahimi: mr.rahimi39@ut.ac.ir
- Reza Karbasi: rezakarbasi@ut.ac.ir
