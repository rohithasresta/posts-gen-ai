import torch
from diffusers import StableDiffusionPipeline
import openai
import re 
import gradio as gr
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from copy import deepcopy
from utils import memeify_image, cleanup_caption
from utils import NEG_PROMPT, OPENAI_TOKEN
import os
from io import BytesIO
CWD = os.getcwd()


MEME_FONT_PATH = os.path.join(CWD,  'fonts', 'impact.ttf')


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "prompthero/openjourney"
PIPE = StableDiffusionPipeline.from_pretrained(MODEL_ID).to(DEVICE)

PROCESSOR = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
MODEL = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")


openai.api_key = OPENAI_TOKEN


def post_generator(task : str, image_desc :str, theme :str, text_pos :str):
    
    image = PIPE(
                  image_desc,
                  negative_prompt=NEG_PROMPT,
                  num_inference_steps=25
                ).images[0]

    content_mapper = {
        'Memes': f"""
                Generate a super funny caption for this:
                Image description: {image_desc}
                Theme: {theme}
                   """,
        'Inspirational Quotes': f"""
                Generate an inspirational quote for this:
                Image description: {image_desc}
                Theme: {theme}
                   """,
        'Slogans': f"""
                Generate an impactfull slogan for this:
                Image description: {image_desc}
                Theme: {theme}
                   """,
        'Jokes': f"""
                Generate a joke for this:
                Image description: {image_desc}
                Theme: {theme}
                   """
    }

    content = content_mapper[task]
    
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "assistant", "content": content}
        ]
    )

    text = response.choices[0]['message']['content']

    text = cleanup_caption(text)

    if text_pos == 'top':
        top = text
        bottom = ''
    else:
        top = ''
        bottom = text

    final_img = memeify_image(image, top=top, bottom=bottom)

    return final_img


def text_generator(task : str, image, theme:str):
    """
    input - image, theme and return text
    """

    inputs = PROCESSOR(image, return_tensors="pt")

    out = MODEL.generate(**inputs)
    image_desc = PROCESSOR.decode(out[0], skip_special_tokens=True)

    content_mapper = {
        "Story": f"""
                Generate a story for this:
                Image description: {image_desc}
                Theme: {theme}
                     """,
        "Poem": f"""
                Generate a poem for this:
                Image description: {image_desc}
                Theme: {theme}
                        """,
        "Lyrics": f"""
                Generate lyrics for this:
                Image description: {image_desc}
                Theme: {theme}
                        """
    }

    content = content_mapper[task]

    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "assistant", "content": content}
        ]
    )

    text = response.choices[0]['message']['content']

    return text
    

def check_inputs_1(selected_option, image_description, theme, text_position):
    if len(str(image_description)) == 0:
        return "Enter image_description"
    elif len(str(theme)) == 0:
        return "Enter Theme"
    elif text_position == None:
        return "Enter Valid text position"
    elif selected_option == None:
        return "Enter Valid Category"
    else:
        return "valid"


def check_inputs_2(selected_option2,image_input,Theme):
    if image_input is None:
        return "Upload Image"
    elif len(str(Theme)) == 0:
        return "Enter Theme"
    elif selected_option2 == None:
        return "Enter Valid Category"
    else:
        return "valid"


def feature1(selected_option, image_description, theme, text_position):
    
    # validating inputs 
    check = check_inputs_1(selected_option, image_description, theme, text_position)
    if check != "valid":
        return {error_box1: gr.update(value=check, visible=True)}
    task = selected_option
    image_desc = image_description 
    theme = theme 
    text_pos = text_position
    
    img = post_generator(task, image_desc, theme,text_pos)

    return [img,error_box]


def feature2(selected_option2,image_input,Theme):
    check = check_inputs_2(selected_option2,image_input,Theme)
    if check != "valid":
        return {error_box: gr.update(value=check, visible=True)}
    task = selected_option2
    image = image_input
    theme = Theme
    text_gen =  text_generator(task , image, theme)

    return [text_gen,error_box]
    

with gr.Blocks() as demo:
    
    gr.Markdown('<style>h1 { background-color: pink; text-align: center; }</style>')  # Added CSS styling for text alignment
    gr.Markdown("<h1>Social Media Post Generator</h1>")  # Centered heading

    with gr.Tab("Text2Post"):
        
        error_box1 = gr.Textbox(label="Error", visible=False)

        """select the option"""
    
        dropdown_options = ["Memes", "Jokes", "Slogans", "Inspirational Quotes"]
        selected_option = gr.inputs.Dropdown(choices=dropdown_options,default="Memes",label="Category")
        
        """inputs """
        image_description = gr.inputs.Textbox(label="Image Description/prompt")
        theme = gr.inputs.Textbox(label="Theme")
        dropdown_two = ["top","bottom"]
        text_position = gr.inputs.Dropdown(choices=dropdown_two,default="top",label="text_position")
        
        """SUBMIT"""
        text_button = gr.Button("SUBMIT")
        
        """OUTPUT"""
        output_image = gr.Image(label="output", width = 512, height = 512)
        
    with gr.Tab("Image2Insight"):
#         with gr.Row():
        error_box = gr.Textbox(label="Error", visible=False)
        dropdown_options = ["Story", "Lyrics","Poem"]
        selected_option2 = gr.inputs.Dropdown(choices=dropdown_options,default="Poem",label="Category")
        Theme = gr.inputs.Textbox(label="Theme")
        image_input = gr.Image()
        image_button = gr.Button("SUBMIT")
        text_output = gr.Textbox(label="output")


    text_button.click(feature1, inputs=[selected_option,image_description, theme, text_position], outputs=[output_image,error_box1])
    image_button.click(feature2,inputs=[selected_option2,image_input,Theme], outputs=[text_output,error_box])

demo.launch(debug = True)
