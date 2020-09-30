import numpy as np
from flask import Flask, render_template, request
import torch
import sys
import os
from PIL import Image
from torchvision import transforms
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = torch.load('tomato_model.pth', map_location=torch.device('cpu'))

def get_class(class_no):
    mappings = {
        '0' : "Bacterial Spot",
        '1' : "Early Blight",
        '2' : "Late Blight",
        '3' : "Leaf Mold",
        '4' : "Septoria Leaf Spot",
        '5' : "Spider mites",
        '6' : "Target Spot",
        '7' : "Tomato yellow leaf curl virus",
        '8' : "Tomato Mosaic virus",
        '9' : "Healthy"
    }
    return mappings[class_no]

def image_transform(imagepath):
    test_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])])
    image = Image.open(imagepath)
    imagetensor = test_transforms(image)
    return imagetensor

def model_predict(img_path, model):
    img = image_transform(img_path)
    img = img.unsqueeze(0)
    model.eval()
    ps = torch.exp(model(img))
    topconf, topclass = ps.topk(1, dim=1)
    return {'class':str(topclass.item()),'confidence':str(topconf.item()*100)}

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        print(basepath)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)
        class_name = get_class(preds['class'])
        if(class_name=='Healthy'):
            return "This is likely to be an healthy tomato plant with {}% of confidence".format(preds['confidence'])
        return "This tomato is likely to have disease {} with {}% of confidence".format(class_name, preds['confidence'])
    return None

# if __name__ == "__main__":
#     app.run(debug=True)