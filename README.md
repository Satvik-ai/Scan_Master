### Scan Master

Scan Master is an optical character recognition tool. It can extract English and Hindi text from the uploaded image. It uses General OCR Theory (GOT), a 580M end-to-end OCR 2.0 model for English optical character recognition and EASYOCR for Hindi optical character recognition. It supports plain text ocr.

It also has a search functionality. User can enter keyword to search within extracted text.

### How to run locally

Open a terminal in the project root directory and run the following commands

1. Install virtualenv:
```
$ pip install virtualenv
```

2. create virtual environment:
```
$ virtualenv env
```

3. Then run the command:
```
$ source env/bin/activate
```

4. Then install the dependencies:
```
$ (env) pip install -r requirements.txt
```

5. Finally start the application:
```
$ (env) python app.py
```

### Live web application
You can also access the live application from the below link:
[Click here]()