# 🏔️Mountain Named Entity Recognition (NER) Project

This project is a **Named Entity Recognition (NER)** system that identifies mountain names in text using a custom-trained model based on the `BERT` architecture.

## 📑Table of Contents
- 📖[Project Overview](#project-overview)
- 🛠️[Installation](#installation)
- 📊 [Dataset Creation](#dataset-creation)
- 🤖[Model](#model)
- 🌐[Hugging Face](#hugging-face)
- 🚀[Usage](#usage)
- 📁[Files](#files)

## 📖Project Overview
This NER project is designed to recognize mountain names within various texts. We leverage the `transformers` library from Hugging Face to train, evaluate, and perform inference using a BERT-based model. The model can be used to highlight mountain names in text for tasks such as automatic labeling of geographic documents or enriching content with additional semantic information.

✨ Key Features:
- Train a BERT model for NER.
- Evaluate the model using confusion matrix, precision-recall, and ROC curve metrics.
- Perform inference on custom texts to identify mountain names.
- Highlight identified mountain names in the output.


## 🛠️Installation
To run this project, you need Python 3.7+ and the required dependencies listed in the `requirements.txt` file.

Clone repository:
```bash
git https://github.com/sofibrezden/Named-Entity-Recognition-Mountains.git
cd Named-Entity-Recognition-Mountains
```

Create a Virtual Environment:
```
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\\Scripts\\activate
```
Install dependencies:
```
pip install -r requirements.txt
```

# 📊Dataset Creation
The dataset was created by:

- **Filtering tokens:** Retained only mountain-related tokens, labeling them as 1, while all other tokens were labeled as 0.
- **Dataset reduction:** Kept all samples containing mountain entities and included a small fraction of non-mountain samples for balance.

# 🤖Model
The model used for this task is bert-base-cased, fine-tuned for Named Entity Recognition (NER) to detect mountain names. It was trained with the following configurations:

- **Number of epochs:** 5
- **Optimizer:** Adam with learning rate of 2e-5
- **Batch size:** 8 (with gradient accumulation)
- **Early stopping:** Applied with a patience of 3 epochs
- **Loss function:** Cross-entropy with token classification.

# 🌐Hugging Face
The trained model and tokenizer are available on Hugging Face:
👉[link](https://huggingface.co/sofibrezden/ner-model-mountains/tree/main)

# 🚀Usage
This project offers two main functionalities: model training and inference.
**1. Model Training:**
You can train the model using the model_training.py script. This script loads the dataset, fine-tunes the BERT model for NER, and saves the trained model.

Training Command:
```
python model_training.py --model-path ./saved_model --epochs 5
```
**2. Model Inference:**
Once the model is trained, you can use it for inference with the `model_inference.py` script. This script loads the trained model from the specified directory and performs NER on the input text.
Inference Command:
```
python model_inference.py --model-path ./saved_model --input-text "I love the Rocky Mountains and Mount Everest."
```
**3. Model Download from Hugging Face:**
Alternatively, if you don't want to train the model, you can download the pre-trained model directly from Hugging Face by specifying:
```
model = AutoModelForTokenClassification.from_pretrained('sofibrezden/ner-model-mountains')
tokenizer = AutoTokenizer.from_pretrained('sofibrezden/ner-model-mountains')
```
# 📁Files
- **dataset_processing.ipynb:** This notebook contains the preprocessing steps for the dataset used in model training. It involves filtering mountain-related tokens, labeling them, and reducing the dataset by keeping all samples with mountain tokens while retaining a small fraction of other samples. The processed dataset is then saved and used in the model training pipeline.
- **model_training.py:** This script handles model training. It loads the dataset, preprocesses the data, and fine-tunes the BERT model for Named Entity Recognition (NER). The trained model is saved in the specified directory.
- **model_inference.py:** Script for performing inference with the trained model. It loads the saved model and tokenizer, processes custom text inputs, and highlights mountain names in the output.
- **demo.ipynb:** Jupyter Notebook that demonstrates the evaluation of the pre-trained model from Hugging Face. It includes model evaluation, inference on test data, and visualizations like the confusion matrix, ROC curve, and precision-recall curve.
- **data/:** This directory stores the training, validation, and test datasets used for model training.
- **improvements_report.pdf:** This file outlines potential improvements for the project.

# 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.