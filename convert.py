from PIL import Image
import os

directory = './train'

for filename in os.listdir(directory):
    if filename.endswith(".png"):
        img = Image.open(os.path.join(directory, filename))
        rgb_img = img.convert('RGB')
        rgb_img.save(os.path.join(directory, filename.replace('.png', '.jpeg')), 'jpeg')
        os.remove(os.path.join(directory, filename))  # delete the original .png file