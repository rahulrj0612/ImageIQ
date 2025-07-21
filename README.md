# ImageIQ

# 🧠 ImageVision-CNN
A clean and customizable **Convolutional Neural Network (CNN)** for image classification, implemented in **PyTorch**. This project is ideal for learning how CNNs work, how to use PyTorch, and how to scale simple models into more complex architectures.

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Features

- 📦 Trainable CNN from scratch using PyTorch
- 🖼️ Custom dataset support (default: CIFAR-10 / MNIST)
- 📉 Training and validation loop with accuracy tracking
- 📊 Matplotlib plots for loss & accuracy
- 💾 Model saving & loading

---

## 🧰 Tech Stack

| Component | Tool       |
|----------|------------|
| Language  | Python 3.9 |
| Framework | PyTorch    |
| IDE/Notebook | Jupyter |
| Plotting | Matplotlib |
| Dataset | CIFAR-10 / MNIST |

---

## 🏗️ Model Architecture

Here's a breakdown of the CNN model:

```
Input → [Conv2D → ReLU → MaxPool] → [Conv2D → ReLU → MaxPool] → Flatten → [FC → ReLU] → [FC → Softmax]
```

In PyTorch:

```python
self.model = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Flatten(),
    nn.Linear(64 * 8 * 8, 128),
    nn.ReLU(),
    nn.Linear(128, 10)  # for CIFAR-10 or 10-class dataset
)
```

---

## 🧪 Training & Evaluation

You can train the model using the notebook interface.

```bash
jupyter notebook Implementing\ CNN\ in\ PyTorch.ipynb
```

Inside the notebook:

- Change the number of epochs, batch size, or optimizer as needed
- Real-time loss/accuracy is printed and plotted
- Model checkpoints are saved using `torch.save`

---

## 📊 Example Outputs

| 📷 Input Image | ✅ Predicted |
|---------------|-------------|
| 🐸 Frog        | Frog        |
| 🚗 Car         | Car         |
| ✈️ Airplane    | Airplane    |

*Add screenshots of loss/accuracy plots here*

---

## 📂 File Structure

```
├── Implementing CNN in PyTorch.ipynb
├── README.md
├── /data (optional)
│   └── custom_dataset/
├── /models
│   └── cnn_model.pth
```

---

## 🧠 Learnings

- Basics of building CNNs
- PyTorch model lifecycle
- DataLoader & Dataset structure
- Model evaluation and inference

---

## 📌 To Do

- [ ] Add dropout or batchnorm
- [ ] Add data augmentation
- [ ] Try deeper networks
- [ ] Add test script for inference on custom images

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙋‍♂️ Contributors

Made with ❤️ by Rahul Raj  
Feel free to [connect](mailto:your-email@example.com) or raise an issue.
