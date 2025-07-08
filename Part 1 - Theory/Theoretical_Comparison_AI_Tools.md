# 🧠 Part 1: Theoretical Understanding

---

## Q1: Main Differences Between TensorFlow and PyTorch

| Feature           | TensorFlow                        | PyTorch                                 |
|-------------------|----------------------------------|-----------------------------------------|
| Developer         | Google                           | Facebook (Meta)                         |
| Syntax Style      | Static computation graph (with eager mode) | Dynamic computation graph (imperative style) |
| Community         | Larger industrial adoption        | Popular in research                     |
| Deployment        | TensorFlow Lite, TensorFlow.js    | TorchServe, ONNX                        |
| Visual Debugging  | TensorBoard (built-in)            | Supports TensorBoard via add-ons         |

**🔍 Use TensorFlow when:**
- You need production-ready deployment (e.g., mobile apps, embedded).
- You're using pre-trained models from TensorFlow Hub.

**🧪 Use PyTorch when:**
- You want fast experimentation or research-oriented workflows.
- You prefer Pythonic, dynamic execution for model debugging.

---

## Q2: Two Use Cases for Jupyter Notebooks in AI Development

**📊 Exploratory Data Analysis (EDA):**
- Load datasets, visualize patterns, and test transformations interactively.
- Example: Examining data distributions and feature correlations.

**🧪 Prototyping and Model Experimentation:**
- Train/test multiple models with live outputs and visualizations (e.g., training curves, confusion matrices).
- Great for collaborative development and reproducibility.

---

## Q3: How spaCy Enhances NLP Tasks vs. Basic Python String Operations

spaCy is a modern NLP library that provides:
- ✅ Pretrained pipelines for tokenization, POS tagging, NER, and lemmatization
- ✅ Accurate models with support for over 60 languages
- ✅ Rule-based patterns and custom entity matcher
- ✅ Faster and more robust than plain string.find or regex for language understanding

**🔍 Basic string operations can't:**
- Detect context-aware entities like organizations or product names.
- Tokenize or analyze language grammatically.
- Handle multilingual text effectively.

---

## Q4: Comparative Analysis — Scikit-learn vs TensorFlow

| Category         | Scikit-learn                        | TensorFlow                              |
|------------------|-------------------------------------|-----------------------------------------|
| Focus            | Classical ML (SVM, Trees, kNN, etc.)| Deep Learning (CNNs, RNNs, Transformers)|
| Learning Types   | Supervised, unsupervised, basic ensemble | Neural networks, custom layers, RL  |
| Complexity       | Simple APIs, great for small datasets| Handles complex models, large data      |
| Beginner-friendly| Very accessible, low learning curve  | Slightly steeper learning curve         |
| Deployment Tools | None built-in                        | TensorFlow Lite, TF Serving             |
| Community + Docs | Mature with tons of tutorials        | Strong backing and rich ecosystem       |

- Use **Scikit-learn** for small-scale tasks like churn prediction or clustering.
- Use **TensorFlow** for image recognition, speech translation, and NLP. 