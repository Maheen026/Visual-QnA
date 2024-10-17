import streamlit as st 
import os
from PIL import Image
import requests
from io import BytesIO
from transformers import ViltForQuestionAnswering, ViltProcessor

st.set_page_config(layout="wide", page_title="VQA")

#Vilt Code
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

def get_answer(image,text):
    try:
        #load and process the image
        img = Image.open(BytesIO(image)).convert('RGB')
        #prepare inputs 
        encoding = processor (img,text,return_tensors="pt")

        #forward pass
        output = model(**encoding)
        logits = output.logits
        idx = logits.argmax(-1).item()
        answer = model.config.id2label[idx]



    except Exception as e:
        return str(e)
st.title("Visual Question Answering Tool")
st.write("Upload an image and enter query to get a response")


col1, col2 = st.columns(2)

#Image Upload
with col1:
    upload_file = st.file_uploader("Upload Image" , type=['jpg','jpeg','png'])
    st.image(upload_file, use_column_width = True)

with col2:
    question = st.text_input("Question")

    if upload_file and question is not None:
        if st.button("Ask query"):
            image = Image.open(upload_file)
            image_byte_array = BytesIO()
            image.save(image_byte_array, format="jpeg")
            image_bytes = image_byte_array.getvalue()

            #get the answer
            answer = get_answer(image_bytes,question)
            st.info("Your Question:" +question)
            st.success("Answer:" +answer)
