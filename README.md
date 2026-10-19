# LivestockModel_Cattely_CIP

LivestockModel starting from Cattely/CIP and evolving.

## Overview

This project is a machine-learning workflow for livestock image recognition, focused on building a gatekeeper-style classifier that can distinguish between valid livestock images and invalid or non-target images. The work starts from the Cattely cattle face dataset and expands into a more general livestock classification setup intended for edge-device deployment.

The notebook is mainly an experimental model-finishing and dataset-preparation pipeline: organizing raw image sources, resizing and standardizing them, training a TensorFlow/Keras classifier, and exporting a compact TFLite model suitable for mobile or edge inference.

## Objective

The goal is to create a practical image recognition model that can:

- identify valid livestock images
- separate cattle face / relevant livestock images from general or invalid captures
- support a lightweight on-device classification workflow
- serve as a foundation for further refinement in a production or research setting

## Data Sources

The notebook combines several data sources:

- Cattely Cattle Face Image Dataset for facial identification / cattle-specific recognition
- CID Cow Images Dataset for general livestock imagery
- Invalid photos such as:
  - cropped or incomplete images
  - farm environment shots
  - human images
  - equipment images
  - accidental or irrelevant captures

This creates a three-class gatekeeper model concept:

- Facial_ID
- General_Cow
- Invalid

## Key Workflow

### 1. Dataset organization

The notebook creates structured data folders under Google Drive for training and validation sets, including:

- raw image folders
- square resized image folders
- class-specific training and validation directories

The raw images are reorganized into consistent folders so they can be consumed by `tf.keras.utils.image_dataset_from_directory`.

### 2. Image preprocessing

Images are standardized to a fixed square size, using a forced resize of 224x224. This supports compatibility with transfer-learning image models such as EfficientNet.

The notebook explicitly prepares a clean dataset pipeline for:

- train data
- validation data
- edge-friendly image dimensions

### 3. Model training

The notebook uses a transfer-learning approach with a Mobile-friendly backbone, specifically EfficientNetB0.

The model structure includes:

- base EfficientNetB0 model
- `GlobalAveragePooling2D()`
- `Dropout(0.2)`
- final output `Dense` layer with softmax activation for class prediction

This is a strong choice for balancing accuracy and model size for edge-device deployment.

### 4. TFLite conversion

After training, the model is converted to TFLite for mobile/embedded deployment. The project also includes an embedder-mode concept: a feature extractor model built from the pooled output of the base model, allowing more compact downstream use cases.

## Why this matters

This repository is a practical example of:

- dataset curation for image recognition
- transfer learning for livestock classification
- image standardization and preprocessing for model input
- model export for on-device or edge inference
- iterative model tuning and experimentation in a notebook-based workflow

It is especially relevant to applications involving:

- livestock identification
- animal monitoring systems
- edge AI for field deployment
- image-gated classification pipelines

## Results and observations

The notebook documents the training process and emphasizes a few important findings:

- the data preparation pipeline was successfully structured for square-image training
- EfficientNetB0 was used as a transfer-learning backbone
- high training accuracy was observed in the early training runs
- validation performance highlighted the need for a more robust validation set and better class separation

One important note from the notebook: a validation dataset containing a single class can produce misleading evaluation results. The model may appear to train well while still lacking meaningful validation performance against real-world distribution. This suggests the need for more diverse validation data and a more balanced label set.

## Next steps

The natural next steps for this work are:

- build a properly balanced validation set across all classes
- measure precision, recall, and confusion matrix by class
- test deployment on a device or mobile runtime
- refine the model for real-world field conditions
- expand the dataset and improve generalization across environments and cattle types

## Repository status

This project is a research and prototype pipeline for livestock image recognition and model tuning, with a strong focus on edge deployment and practical computer vision workflows.

## License

This repository is shared for learning, experimentation, and portfolio use unless otherwise specified.

## Profile summary

Software engineering, machine learning, network applications, and GPU programming.
