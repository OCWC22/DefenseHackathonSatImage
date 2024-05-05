import argparse
import torch
import time
import sys
import gradio as gr
sys.path.append('../moondream')
from moondream import detect_device, LATEST_REVISION
from threading import Thread
from transformers import TextIteratorStreamer, AutoTokenizer, AutoModelForCausalLM
from PIL import ImageDraw
import re
from torchvision.transforms.v2 import Resize

parser = argparse.ArgumentParser()
parser.add_argument("--cpu", action="store_true")
args = parser.parse_args()

if args.cpu:
    device = torch.device("cpu")
    dtype = torch.float32
else:
    device, dtype = detect_device()
    if device != torch.device("cpu"):
        print("Using device:", device)
        print("If you run into issues, pass the `--cpu` flag to this script.")
        print()

model_id = "vikhyatk/moondream2"
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=LATEST_REVISION)
moondream = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=LATEST_REVISION
)
path = "checkpoints/float"
moondream.vision_encoder.projection.load_state_dict(
    torch.load(f"{path}/vision_projection.final.pt", map_location="cpu")
)
moondream.vision_encoder.encoder.load_state_dict(
    torch.load(f"{path}/vision_encoder.final.pt", map_location="cpu")
)
moondream.text_model.load_state_dict(
    torch.load(f"{path}/text_model.final.pt", map_location="cpu")
)
moondream = moondream.to(device=device, dtype=dtype)
moondream.eval()


def answer_question(img, prompt):
    image_embeds = moondream.encode_image(img)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    thread = Thread(
        target=moondream.answer_question,
        kwargs={
            "image_embeds": image_embeds,
            "question": prompt,
            "tokenizer": tokenizer,
            "streamer": streamer,
        },
    )
    thread.start()

    buffer = ""
    for new_text in streamer:
        buffer += new_text
        yield buffer


def extract_ints(text):
    # Regular expression to match an array of exactly four integers
    pattern = r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]"

    # Search for the pattern
    match = re.search(pattern, text)
    if match:
        # Convert matched strings to integers
        return [int(num) * 14.0 / 378 for num in match.groups()]
    return None  # Return None if no match is found


def extract_floats(text):
    # Regular expression to match an array of four floating point numbers
    pattern = r"\[\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\]"
    match = re.search(pattern, text)
    if match:
        # Extract the numbers and convert them to floats
        return [float(num) for num in match.groups()]
    return None  # Return None if no match is found


def extract_bbox(text):
    bbox = None
    if extract_ints(text) is not None:
        x1, y1, x2, y2 = extract_ints(text)
        bbox = (x1, y1, x2, y2)
    if extract_floats(text) is not None:
        x1, y1, x2, y2 = extract_floats(text)
        bbox = (x1, y1, x2, y2)
    return bbox


def process_answer(img, answer):
    if extract_bbox(answer) is not None:
        x1, y1, x2, y2 = extract_bbox(answer)
        draw_image = Resize(768)(img)
        width, height = draw_image.size
        x1, x2 = int(x1 * width), int(x2 * width)
        y1, y2 = int(y1 * height), int(y2 * height)
        bbox = (x1, y1, x2, y2)
        ImageDraw.Draw(draw_image).rectangle(bbox, outline="red", width=3)
        return gr.update(visible=True, value=draw_image)

    return gr.update(visible=False, value=None)


with gr.Blocks() as demo:
    gr.Markdown("# 🌔 moondream")

    gr.HTML(
        """
        <style type="text/css">
            .md_output p {
                padding-top: 1rem;
                font-size: 1.2rem !important;
            }
        </style>
        """
    )

    with gr.Row():
        prompt = gr.Textbox(
            label="Prompt",
            value="What's going on? Respond with a single sentence.",
            interactive=True,
        )
    with gr.Row():
        img = gr.Image(type="pil", label="Upload an Image", streaming=True)
        with gr.Column():
            output = gr.Markdown(elem_classes=["md_output"])
            ann = gr.Image(visible=False, label="Annotated Image")

    latest_img = None
    latest_prompt = prompt.value

    @img.change(inputs=[img])
    def img_change(img):
        global latest_img
        latest_img = img

    @prompt.change(inputs=[prompt])
    def prompt_change(prompt):
        global latest_prompt
        latest_prompt = prompt

    @demo.load(outputs=[output])
    def live_video():
        while True:
            if latest_img is None:
                time.sleep(0.1)
            else:
                for text in answer_question(latest_img, latest_prompt):
                    if len(text) > 0:
                        yield text

    output.change(process_answer, [img, output], ann, show_progress=False)

demo.queue().launch(debug=True)
