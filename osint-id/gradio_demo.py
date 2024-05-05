import argparse
import torch
import gradio as gr
import sys
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
path = "checkpoints/grid"
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
    gr.Markdown(
        """
        # 🌔 moondream
        """
    )
    with gr.Row():
        prompt = gr.Textbox(label="Input Prompt", placeholder="Type here...", scale=4)
        submit = gr.Button("Submit")
    with gr.Row():
        img = gr.Image(type="pil", label="Upload an Image")
        with gr.Column():
            output = gr.Markdown(label="Response")
            ann = gr.Image(visible=False, label="Annotated Image")

    submit.click(answer_question, [img, prompt], output)
    prompt.submit(answer_question, [img, prompt], output)
    output.change(process_answer, [img, output], ann, show_progress=False)

demo.queue().launch(debug=True)
