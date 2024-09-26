import gradio as gr
import spaces
from transformers import AutoModel, AutoTokenizer
import os
import base64
import io
import uuid
import time
import shutil
from pathlib import Path
import re
import easyocr

# OCR Model
tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True)
model = model.eval().cuda()
reader = easyocr.Reader(['hi'])

UPLOAD_FOLDER = "./uploads"
RESULTS_FOLDER = "./results"

for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# OCR Processing of the image uploaded by the user
@spaces.GPU
def run_GOT(image,language):
    unique_id = str(uuid.uuid4())
    image_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}.png")
    
    shutil.copy(image, image_path)
    
    try:
        if language == "English":
            res = model.chat(tokenizer, image_path, ocr_type='ocr')
            return res
        elif language == "Hindi":
            res = reader.readtext(image)
            extracted_text = ''
            for x in res:
                extracted_text += x[1] + '\n'
            return extracted_text
        else:
            english_extraction = model.chat(tokenizer, image_path, ocr_type='ocr')
            hindi_extraction = reader.readtext(image)
            hindi_extract = ''
            for x in hindi_extraction:
                hindi_extract += x[1] + '\n'
            return english_extraction+'\n'+hindi_extract
    except Exception as e:
        return f"Error: {str(e)}", None
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

# Search Functionality
def search_keyword(text,keyword):
    # Convert text and keyword to lowercase for case-insensitive search
    text_lower = text.lower()
    keyword_lower = keyword.lower()

    # Keyword position in the text
    pos = text_lower.find(keyword_lower)

    if pos == -1:
        ans = '<h3 style="text-align: center;">'+"Keyword not found"+'</h3>'
    else:
        res = [i.start() for i in re.finditer(keyword_lower, text)]
        ans = '<h3>'
        l = 0
        for x in res:
            ans += text[l:x]+'<mark>'+text[x:x+len(keyword)]+'</mark>'
            l += len(text[l:x]+text[x:x+len(keyword)])
        ans += text[l:]+'</h3>'
    return ans

def cleanup_old_files():
    current_time = time.time()
    for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]:
        for file_path in Path(folder).glob('*'):
            if current_time - file_path.stat().st_mtime > 3600:  # 1 hour
                file_path.unlink()

title_html = """
<h1> <span class="gradient-text" id="text">Scan Master</span></h1>
<p>Scan Master uses General OCR Theory (GOT), a 580M end-to-end OCR 2.0 model for English optical character recognition and EASYOCR for Hindi optical character recognition. It supports plain text ocr.</p>
"""

acknowledgement_html = """
<h3>Acknowledgement</h3>
<a href="https://huggingface.co/ucaslcl/GOT-OCR2_0">[😊 Hugging Face]</a> 
<a href="https://arxiv.org/abs/2409.01704">[📜 Paper]</a>
<a href="https://github.com/Ucas-HaoranWei/GOT-OCR2.0/">[🌟 GitHub]</a> 
"""

aboutme_html = """
<h3>About Me</h3>
<p>Name : Satvik Chandrakar</p>
<a href="https://github.com/Satvik-ai">[🌟 GitHub]</a> """


# Scan Master web application developed using Gradio
with gr.Blocks() as scan_master_web_app:
    gr.HTML(title_html)
    gr.Markdown("""
    You need to upload your image below and choose appropriate language, then click "Submit" to run the model. More characters will result in longer wait times.""")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="filepath", label="Upload your image")
            gr.Markdown("""If your image contains only English text, then choose English option in the language. If it contains only Hindi text, then choose Hindi option in the language. If it contains both the language, then choose the third option.""")
            lang_dropdown = gr.Dropdown(
                choices=[
                    "English",
                    "Hindi",
                    "English + Hindi",
                ],
                label="Choose language",
                value="English"
            )
            submit_button = gr.Button("Submit")
        
        with gr.Column():
            ocr_result = gr.Textbox(label="GOT output")
    
    with gr.Row():
        with gr.Column():
            keyword = gr.Textbox(label="Search a keyword in the extracted text")
            search_button = gr.Button("Search")
        
        with gr.Column():
            search_result = gr.HTML(label="Search result")
    
    gr.HTML(acknowledgement_html)
    gr.HTML(aboutme_html)

    submit_button.click(
        run_GOT,
        inputs=[image_input,lang_dropdown],
        outputs=[ocr_result]
    )

    search_button.click(
        search_keyword,
        inputs=[ocr_result,keyword],
        outputs=[search_result]
    )

if __name__ == "__main__":
    cleanup_old_files()
    scan_master_web_app.launch()