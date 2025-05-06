# 🖨️ Scan Master

**Scan Master** is an optical character recognition (OCR) tool capable of extracting both **English** and **Hindi** text from uploaded images. It supports plain text extraction and includes a **search** functionality that lets users look for specific keywords within the extracted content.

---

## 🧠 Technologies & Models

- **English OCR**: [GOT (General OCR Theory)](https://arxiv.org/abs/2409.01704) – a 580M parameter end-to-end OCR 2.0 model.
- **Hindi OCR**: [EasyOCR](https://www.jaided.ai/easyocr/) – a multilingual OCR library.
- **Framework**: Flask (Python)
- **Search Capability**: Keyword search on extracted text.

---

## 🚀 Features

- Upload image and extract English or Hindi text.
- Plain text OCR support.
- Keyword-based search within extracted results.
- Lightweight and easy to deploy locally or via web.

---

## 💻 How to Run Locally

Open a terminal in the project root directory and run the following commands

1. Install virtualenv:
```
$ pip install virtualenv
```

2. create virtual environment:
```
$ virtualenv env
```

3. Then run the following command (for windows):
```
$ .\env\Scripts\activate
```

4. Install Torch with Cuda enabled:
```
$ (env) pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

5. Then install the other dependencies:
```
$ (env) pip install -r requirements.txt
```

6. Finally start the application:
```
$ (env) python app.py
```

---

## 🌐 Live Demo
You can also access the live application from the below link:
[Click here](https://huggingface.co/spaces/Satvik-ai/Scan_Master)

---

## 🙏 Acknowledgements
- [😊 Hugging Face](https://huggingface.co/ucaslcl/GOT-OCR2_0)
- [📜 Paper](https://arxiv.org/abs/2409.01704)
- [🌟 GitHub](https://github.com/Ucas-HaoranWei/GOT-OCR2.0/)
