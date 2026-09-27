# Bird vs. Forest Classifier

A small end-to-end image classification project: fine-tunes a pretrained
**ResNet18** to tell photos of **birds** apart from photos of **forests**.

Built as a learning project for transfer learning with PyTorch — dataset
collection, two-stage fine-tuning, and inference are all in a single
notebook.

## Pipeline

1. **Download images** — searches DuckDuckGo (`ddgs`) for bird/forest photos
   and downloads them directly, verifying each one is a valid image before
   saving.
2. **Clean the dataset** — removes any corrupt/truncated downloads, then
   sanity-checks that both classes have enough images and aren't badly
   imbalanced before continuing.
3. **Train/val split** — 80/20 split via `torchvision.datasets.ImageFolder`.
4. **Two-stage fine-tuning**:
   - *Stage 1* — freeze the pretrained backbone, train only the new
     classification head.
   - *Stage 2* — unfreeze the whole network and fine-tune end-to-end at a
     lower learning rate.
5. **Save / reload** the trained model (`.pth` state dict).
6. **Visualize** a batch of training images with their labels.
7. **Run predictions** on a new image — either a local file path or an
   image URL.

## Requirements

```bash
pip install torch torchvision ddgs pillow numpy matplotlib requests
```

For GPU training, install the CUDA build of PyTorch matching your setup —
see [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).
CPU-only also works fine for this project's small dataset size.

## Usage

Open `bird_or_forest_classifier.ipynb` and run the cells top to bottom.

- All key settings (image size, batch size, epochs, dataset size, paths) are
  in the config block in the first code cell.
- Set `FRESH_DOWNLOAD = True` in the download cell to wipe and re-download
  the dataset from scratch; set it to `False` to reuse whatever's already in
  the data folder.
- Set `USE_COLAB_DRIVE = True` if running in Google Colab and you want the
  trained model persisted to Google Drive; leave it `False` to save locally
  next to the notebook.

Runs both locally and in Google Colab.

## Notes

- The dataset is scraped on the fly rather than bundled, so results vary
  run to run and depend on what image search returns that day. For anything
  beyond a learning/demo project, swapping in a fixed, curated dataset
  (e.g. from Kaggle) is recommended.
- A notebook cell will raise an assertion error if one class ends up with
  too few images or the classes are too imbalanced (a common failure mode
  with live scraping) — re-run the download cell if that happens.

## License

Add a license of your choice (e.g. MIT) if you plan to share this publicly.
