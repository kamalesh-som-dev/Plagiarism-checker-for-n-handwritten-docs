from flask import Flask, render_template, request
import boto3
import fitz
import io
from PIL import Image, ImageDraw
import os
import re
import math
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

app = Flask(__name__)

pdf_folder="PDF_data"
app.config['UPLOAD_FOLDER'] = 'PDF_data'
output_folder = "Temp"
txt_folder="txt_data"

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


#plagiarism checker
def preprocess_text(text):
    # Convert text to lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def calculate_cosine_similarity(text1, text2):
    # Tokenize and preprocess the texts
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)
    # Create TF-IDF vectors for the documents
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    # Calculate the cosine similarity between the two vectors
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return similarity

def check_plagiarism(file1_path, file2_path, similarity_threshold=0.7):
    # Read the content of the two files
    with open(file1_path, 'r', encoding='latin-1') as file1:
        document1 = file1.read()
    with open(file2_path, 'r', encoding='latin-1') as file2:
        document2 = file2.read()
    # Calculate the cosine similarity between the documents
    similarity = calculate_cosine_similarity(document1, document2)
    if similarity >= similarity_threshold:
        return f"\nPlagiarism detected with Similarity score: {similarity:.2f}\n"
    else:
        return f"\nNo plagiarism detected with Similarity score: {similarity:.2f}\n"

#preprocessing pdfs
def pdf_to_images(pdf_path):
    images = []
    pdf_document = fitz.open(pdf_path)
    for page_number in range(pdf_document.page_count):
        page = pdf_document.load_page(page_number)
        img_list = page.get_images(full=True)
        for img_info in img_list:
            base_image = pdf_document.extract_image(img_info[0])
            image_data = base_image["image"]
            image = Image.open(io.BytesIO(image_data))
            images.append(image)
    pdf_document.close()
    return images

def convert_to_grayscale(image):
    return image.convert("L")

def main(pdf_path, output_folder):
    images = pdf_to_images(pdf_path)
    for i, image in enumerate(images):
        grayscale_image = convert_to_grayscale(image)
        output_path = f"{output_folder}/image_{i + 1}.jpg"
        grayscale_image.save(output_path)
    return images

#reduce api stress, api works!
def extract_text_from_pdf(pdf_file_path,txt_name):
    txt_file_names=[]
    images=main(pdf_file_path, output_folder)
    client = boto3.client('textract',region_name='',aws_access_key_id='',aws_secret_access_key='')
    full_txt=""
    for i,image in enumerate(images):
        ip_path=  f"Temp/image_{i + 1}.jpg"
        image=Image.open(ip_path)
        with open(ip_path, 'rb') as image:
            img = bytearray(image.read()
        )
        response = client.detect_document_text(
            Document={'Bytes': img}
        )
        text = ""
        for item in response["Blocks"]:
            if item["BlockType"] == "LINE":
                text = text + " "+item["Text"]
        full_txt=full_txt+text+"\n"
    files = os.listdir(output_folder)
    for file in files:
        file_path = os.path.join(output_folder, file)
        # Check if the path is a file (not a directory)
        if os.path.isfile(file_path):
            # Delete the file
            os.remove(file_path)
    file_path = os.path.join(txt_folder, f"{txt_name}.txt")
    txt_file_names.append(txt_name+".txt")
    with open(file_path, 'a') as file:  # 'w' for writing (creates a new file or overwrites an existing file)
        file.write(full_txt)
    return full_txt

@app.route('/', methods=['GET', 'POST'])
def upload_files():
    files = request.files.getlist('files[]')
    if files:
        texts = []
        file_paths = []

        # Save uploaded files and extract text from each PDF
        for i, file in enumerate(files):
            # Generate a unique file name for each uploaded PDF
            file_name = f'file{i + 1}.pdf'
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
            file.save(file_path)
            file_paths.append(file_path)
        # 
        # Extract text and create similarity matrix
        texts = []
        for file_path in file_paths:
            text = extract_text_from_pdf(file_path, os.path.splitext(os.path.basename(file_path))[0])
            texts.append(text)
            print(os.path.basename(file_path))
        similarity_matrix, files = create_similarity_matrix_from_folder(txt_folder)
        return render_template('index.html', similarity_matrix=similarity_matrix, files=files)
    return render_template('index.html', similarity_matrix=None,files=None)

def read_text_from_folder(folder_path):
    texts = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='latin-1') as file:
                text = file.read()
                texts[filename] = text
    return texts

def create_similarity_matrix_from_folder(folder_path):
    text_files = read_text_from_folder(folder_path)
    files = list(text_files.keys())
    matrix = []
    for i in range(len(files)):
        row = []
        for j in range(len(files)):
            if i != j:
                similarity_score = calculate_cosine_similarity(text_files[files[i]], text_files[files[j]])
                row.append(round(similarity_score,2))
            else:
                row.append(None)
        matrix.append(row)
    return matrix, files

if __name__ == '__main__':
    app.run(debug=True)