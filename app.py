import numpy as np
from flask import Flask, render_template, request
import torch

app = Flask(__name__)

@app.route('/')
def home():
    return "<h1>Hello</h1>"


if __name__ == "__main__":
    app.run(debug=True)