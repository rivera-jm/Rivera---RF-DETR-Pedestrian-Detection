# CO2.2 — RF-DETR Nano Pedestrian Detection

A Google Colab implementation of single-class pedestrian object detection using the official **RF-DETR Nano** model. The project reuses a YOLO-format pedestrian dataset prepared in the CO1.1–CO1.4 activities and evaluates the fine-tuned model on a held-out test split.

## Repository contents

- `CO2_2_RF_DETR_Pedestrian_Detection_GitHub.ipynb` — commented, step-by-step Python/Colab implementation of dataset checks, training, evaluation, IoU/matched-class calculations, and visualizations.
- `README.md` — setup, dataset format, and reproducibility notes.

Notebook outputs were cleared for a smaller GitHub upload. The completed experimental outputs and trained checkpoints were saved separately in the author's Google Drive; **the repository notebook itself does not contain the images, checkpoints, or original training outputs**.

## Dataset

- Object class: `person` (YOLO class ID `0`)
- Images: **888** (639 train, 119 validation, 130 test)
- Bounding-box annotations: **4,366** (3,303 train, 467 validation, 596 test)
- Format: YOLO normalized `class x_center y_center width height`
- Source: existing CO1.1–CO1.4 pedestrian-detection dataset, originally prepared and annotated using Roboflow. The chronological split used here is a locally prepared derivative. **A public download URL for this exact split has not yet been provided.**

The expected Google Drive structure for the notebook is:

```text
MyDrive/
└── CO1_1_Pedestrian_Detection/
    ├── chronological_dataset/
    │   ├── train/{images,labels}/
    │   ├── valid/{images,labels}/
    │   ├── test/{images,labels}/
    │   └── data.yaml
    └── CO1_3_Ground_Truth/
        └── corrected_test_labels/
```

Example `chronological_dataset/data.yaml`:

```yaml
names:
  - person
nc: 1
train: train/images
val: valid/images
test: test/images
```

**Important:** Cells 1–8 inspect the existing CO1.1–CO1.4 project, including some directories beyond the chronological dataset. For a fresh setup, either reproduce that directory layout or adapt those earlier inspection cells. The remaining cells use paths rooted at `/content/drive/MyDrive/CO1_1_Pedestrian_Detection`; change `DATASET_ROOT` in Cell 2 if your Google Drive path differs. Cell 5/7 requires the historical annotation-review files, which are for documenting prior work rather than essential RF-DETR training inputs. Cell 16 assumes exactly 130 test images.

## Environment and setup

The reported run used Google Colab with an NVIDIA Tesla T4 GPU, Python 3.13.15, PyTorch 2.11.0+cu130, and `rfdetr` 1.11.2.

1. Open the notebook in **Google Colab**.
2. Select **Runtime → Change runtime type → GPU**.
3. Put the dataset in Google Drive using the structure above and update the Google Drive root path if needed.
4. Run the cells in notebook order. Cell 10 installs the training package using `pip install 'rfdetr[train]'` (other required Python packages are imported by the notebook).
5. **Cell 13 trains the model** and saves checkpoints under `CO1_1_Pedestrian_Detection/CO2_2_RF_DETR/training_results/`. Do not run it if you only intend to review code or results.
6. **Cells 14–19** load the saved best checkpoint, run independent test evaluation, generate prediction overlays, compute additional IoU-based metrics, and create performance plots. These require the trained checkpoint and original test labels.

Training is managed by the official RF-DETR package, rather than an independently reimplemented transformer or loss function.

## Training configuration

| Parameter | Value |
| --- | --- |
| Model | Pretrained RF-DETR Nano |
| Epochs | 20 |
| Batch size | 2 |
| Gradient accumulation | 4 |
| Nominal effective batch size | 8 |
| Learning rate | 0.0001 |
| Early stopping patience | 5 |
| Checkpoint interval | 5 |
| Best-checkpoint selection | Validation mAP |

The run reached the maximum 20 epochs. Training loss declined from **5.7183** at the first recorded epoch to **4.4903** at the final epoch.

## Held-out test results

| Metric | Value |
| --- | ---: |
| mAP@0.50 | 0.6488 |
| mAP@0.50:0.95 | 0.3855 |
| mAP@0.75 | 0.3749 |
| Precision (RF-DETR F1 sweep) | 0.8771 |
| Recall (RF-DETR F1 sweep) | 0.5268 |
| F1 (RF-DETR F1 sweep) | 0.6583 |
| Mean average recall | 0.5221 |

Additional one-to-one IoU-based evaluation at **confidence 0.50** and **IoU 0.50** produced **290 TP**, **24 FP**, **306 FN**, **precision 0.9236**, **recall 0.4866**, and **mean matched IoU 0.8300**. Accuracy over valid matched detections was **1.0000**, but this is a *single-class dataset* and does **not** mean that all pedestrians were detected. The two precision/recall calculations use different thresholds/protocols and should not be conflated.

## Limitations

- The 130 test images are from a video-derived chronological split; other video streams or environments were not evaluated.
- The classifier has only one class (`person`), limiting the interpretability of classification accuracy.
- The reported fixed-threshold metrics show numerous missed pedestrians (306 FN).
- The repository notebook uses the original author's Google Drive layout and is **not fully independently executable without the corresponding dataset and trained checkpoint**.
- The activity describes RF-DETR as a receptive-field-enhanced DETR architecture; this implementation instead uses **Roboflow's official RF-DETR Nano** package and does not claim a separately implemented receptive-field module.

## Acknowledgment

Model implementation: [RF-DETR official repository](https://github.com/roboflow/rf-detr). Dataset origin and redistribution permissions should be verified before publishing original or processed images.
