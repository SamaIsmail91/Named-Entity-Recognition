<div align="center">

  <h1>🔍 Named Entity Recognition (NER) Studio</h1>
  <p><b>An End-to-End Comparative Study: From Classical Recurrent Models (LSTM / BiLSTM / CRF) to Fine-Tuned Transformers (DistilBERT)</b></p>

  [![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
  [![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)
  [![Gradio](https://img.shields.io/badge/Gradio-FF7C00?logo=gradio&logoColor=white)](https://gradio.app/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📌 Project Overview

This repository presents a complete pipeline for **Named Entity Recognition (NER)** on standard sequence tagging benchmarks. The core objective is to analyze, benchmark, and compare traditional deep learning sequence models built from scratch against state-of-the-art pretrained transformer architectures.

The system is deployed via an interactive **Gradio Studio UI**, allowing users to visually inspect and highlight entities (`PERSON`, `ORGANIZATION`, `LOCATION`, `MISC`) across different model architectures in real-time.

---

## 🛠️ Key Features

- **Multi-Model Support:** Implementations spanning from scratch-built recurrent networks to fine-tuned transformer layers.
- **Custom Alignment Logic:** Robust subword-to-label alignment (`tokenize_and_align_labels`) ensuring exact prediction mapping for sub-tokenized inputs.
- **Interactive Web App:** A custom-styled Gradio dashboard featuring dynamic entity highlighting and entity aggregation statistics.
- **Comprehensive Evaluation:** Strict entity-level evaluation leveraging `seqeval` metrics (Precision, Recall, F1-Score).

---

## 🏗️ Architecture Comparison

| Model Architecture | Type | Pretrained | Description |
| :--- | :---: | :---: | :--- |
| **LSTM** | Baseline | ❌ No | Standard unidirectional LSTM trained from scratch on character/word embeddings. |
| **BiLSTM** | Recurrent | ❌ No | Bidirectional LSTM capturing both left and right contextual representations. |
| **BiLSTM + CRF** | Hybrid | ❌ No | Combines BiLSTM features with a **Conditional Random Field (CRF)** decoding layer for joint tag sequence optimization. |
| **Fine-tuned Transformer** | Transformer | ✅ Yes | **DistilBERT** encoder with a token classification head fine-tuned for high-accuracy NER. |

---

## 📊 Dataset & Named Entities

The models are trained and evaluated on CoNLL-style sequential tags using the **IOB2 tagging scheme**:

- 👤 **PER** (Person): *Barack Obama, Marie Curie*
- 🏢 **ORG** (Organization): *Google, United Nations*
- 📍 **LOC** (Location): *Berlin, Paris, France*
- 🎨 **MISC** (Miscellaneous): *G7 Summit, Nobel Prize*

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/your-username/ner-studio.git](https://github.com/your-username/ner-studio.git)
cd ner-studio
pip install -r requirements.txt
Requirements: torch, transformers, datasets, seqeval, gradio, pandas, numpy

2. Run the Interactive Studio
Launch the Gradio application locally:

Bash
python app.py
Access the studio in your browser at http://localhost:7860.

📈 Evaluation & Performance
Models were evaluated on the held-out test split using standard entity-level F1 metrics via seqeval:

Plaintext
               precision    recall  f1-score   support

         LOC       0.91      0.93      0.92      1668
        MISC       0.82      0.85      0.83       702
         ORG       0.86      0.88      0.87      1661
         PER       0.94      0.96      0.95      1617

   micro avg       0.89      0.92      0.91      5648
   macro avg       0.88      0.90      0.89      5648
💻 Visual Demo
Here is a glimpse of the Gradio interface highlighting entities in real-time:

Plaintext
[ Input Text ] -> "Barack Obama met Angela Merkel in Berlin."
[ Highlighted Output ] -> [Barack Obama](PER) met [Angela Merkel](PER) in [Berlin](LOC).
🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

📜 License
Distributed under the MIT License. See LICENSE for more information.


---

### 💡 نصائح إضافية لتخصيص الملف:
1. استبدلي `your-username/ner-studio` برابط الـ GitHub الخاص بكِ.
2. يمكنكِ إضافة صورة متحركة (GIF) أو Screenshot للـ Gradio App بعد تشغيله تحت قسم **Visual Demo**.




