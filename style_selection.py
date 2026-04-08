from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import cv2

import torch
from PIL import Image
import open_clip
import os

def load_detector():
    cfg = get_cfg()
    cfg.MODEL.DEVICE='cpu'
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.80

    return DefaultPredictor(cfg)


def detect_clothing(image_path, predictor):
    image = cv2.imread(image_path)
    outputs = predictor(image)

    boxes = outputs["instances"].pred_boxes
    cropped_images = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.tolist())
        crop = image[y1:y2, x1:x2]
        if crop.size > 0:
            cropped_images.append(crop)

    return cropped_images

# =========================
# 4. clip_model.py (CLIP Embeddings)
# =========================
import torch
#import clip
from PIL import Image

DB_FOLDER = "db"
os.makedirs(DB_FOLDER, exist_ok=True)

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer('ViT-B-32')


def load_clip():

    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    #model, preprocess = clip.load("ViT-B/32", device=device)


    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    return model, preprocess, device


def get_image_embedding(img_array, model, preprocess, device):

    image = Image.fromarray(img_array)
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encode_image(image)

    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().flatten()


def get_text_embedding(text, model, device):

    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    #tokens = clip.tokenize([text]).to(device)

    tokens = tokenizer([text]).to(device)

    with torch.no_grad():
        emb = model.encode_text(tokens)

    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().flatten()

# =========================
# 5. inference.py (Detection + CLIP Search)
# =========================
import faiss
import numpy as np
#from detectron import load_detector, detect_clothing
#from clip_model import load_clip, get_image_embedding, get_text_embedding

from collections import OrderedDict
from deepface import DeepFace

predictor = load_detector()
clip_model, preprocess, device = load_clip()

alpha = 20

def create_embeddings(embeddings):

    embeddings = np.array(embeddings).astype("float32")
    dim = len(embeddings[0])

    # added for cosine similarity
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index

def build_index(image_dict,gender_array):


    image_arrays = list(image_dict.values())

    # combine gender and image embeddings
    image_embeddings = [
        np.concatenate( [ get_image_embedding(img, clip_model, preprocess, device), alpha * gender_to_vector( gender_array[i]) ])  for i, img in enumerate(image_arrays) ]

    text_embeddings = [ get_image_embedding(img, clip_model, preprocess, device) for img in image_arrays ]

    text_index = create_embeddings(text_embeddings)
    image_index = create_embeddings(image_embeddings)

    return text_index, image_index


# -------------------------------
# 3. Gender Detection (DeepFace)
# -------------------------------
def predict_gender_from_crop(crop):
    # Save temporary image
    #temp_path = "/content/temp_crop.jpg"
    #cv2.imwrite(temp_path, crop)

    result = DeepFace.analyze(
        img_path=crop,
        actions=['gender'],
        detector_backend='retinaface',
        enforce_detection=False
    )

    gender_scores = result[0]['gender']
    gender = max(gender_scores, key=gender_scores.get)

    return gender, gender_scores

def gender_to_vector(gender):
    print(gender)
    if gender == "Man":
        return np.array([1.0, 0.0], dtype="float32")
    elif gender == "Woman":
        return np.array([0.0, 1.0], dtype="float32")
    else:
        return np.array([0.0, 0.0], dtype="float32")


def search_by_image(query_path, index, dataset_images, k=2):

    crops = detect_clothing(query_path, predictor)

    results = []
    for crop in crops:

        q_emb = get_image_embedding(crop, clip_model, preprocess, device)
        gender, gender_scores = predict_gender_from_crop( crop ) 

        gender_vec = gender_to_vector(gender)

        gender_vec = alpha * gender_vec

        final_embedding = np.concatenate([q_emb, gender_vec]).reshape(1,-1)

        #final_embedding = final_embedding / np.linalg.norm(final_embedding)

        # added for cosine similarity
        faiss.normalize_L2(final_embedding)

        D, I = index.search(final_embedding, k)
        results.extend([list(dataset_images.keys())[i] for i in I[0]])

    return results
    #return I[0]


def search_by_text(query_text, index, dataset_images, k=2):

    q_emb = get_text_embedding(query_text, clip_model, device).reshape(1, -1)

    D, I = index.search(q_emb, k)

    return [list(dataset_images.keys())[i] for i in I[0]]
    #return I[0]


def load_dataset(folder):

    images = OrderedDict()
    paths = []

    gender_array = []

    #for file in os.listdir(folder):
    for file in folder:
        #path = os.path.join(folder, file)
        img = cv2.imread(file)

        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images[file] = img
            #paths.append(path)

            gender, gender_scores = predict_gender_from_crop( img ) 
            gender_array.append(gender)

    return images, gender_array

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

from flask import Flask,render_template_string
from threading import Thread
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from flask import send_from_directory

import shutil

import os

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# 📁 Upload folder
DB_FOLDER = "db"

os.makedirs(DB_FOLDER, exist_ok=True)
app.config["DB_FOLDER"] = DB_FOLDER

files = ["IMG_6414.jpg","20211121_205340.jpg","IMG_6545.jpg","IMG_6526.jpg","IMG_6587.jpg","IMG_6526.jpg"]


files = files + ["j1.jpg","j2.jpg","f1.jpg","f2.jpg","j4.jpg"]

files = files + ["wm1.jpg", "wm2.jpg", "wm3.jpg", "w1.jpg","w2.jpg","w3.jpg","w5.jpg","w7.jpg","jn1.jpg","jn2.jpg","jn3.jpg","jn4.jpg","jn6.jpg"]

for f in files:
  shutil.copy("/content/" + f, "db/"+f)

image_dataset, gender_array = load_dataset(files)

text_index, image_index = build_index(image_dataset,gender_array)

# 🔹 HTML UI served directly (no templates folder needed)

UPLOAD_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Upload File</title>
</head>
<body>
    <h2>Enter Search Query and Upload Desired Image</h2>

    <form action="/post" method="POST" enctype="multipart/form-data">
        
        <input type="text" name="search_query" placeholder="Enter Search Query">
        <br><br>
        <input type="file" name="file" placeholder="Upload Image to Search Against">
        <br><br>
        <button type="submit">Submit</button>
    </form>

</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(UPLOAD_PAGE)

@app.route('/db/<filename>')
def db_file(filename):
    return send_from_directory(app.config["DB_FOLDER"], filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# 🔹 POST request
@app.route('/post', methods=['POST'])
def upload_file():


    if request.method == 'GET':
        return "Send POST request with file"

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    #query = request.text['search_query']
    query = request.form.get('search_query')
    text_images = search_by_text(query, text_index, image_dataset)

    file = request.files['file']

    filename = secure_filename(file.filename)

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    style_images = search_by_image(filepath, image_index, image_dataset)


    return render_template_string("""
        <h2>Text Search Results</h2>

        <h1> Entered Query : {{query}} </h1> <br> <br>
        {% for img in text_images %}
            <img src="/db/{{img}}" width="300">
        {% endfor %}
        <h2> Style Search Results </h2>
        <h1> Query Style Image :</>
          <img src="/uploads/{{filename}}" width="300">  <br <br>
        {% for img in style_images %}
            <img src="/db/{{img}}" width="300">
        {% endfor %} 
    """, text_images= text_images, style_images = style_images,query = query,filename = filename)

    return jsonify({
        "text_images": text_images,
        "style_images": style_images
    })




def run():
    app.run(host='0.0.0.0', port=5000)

Thread(target=run).start()

