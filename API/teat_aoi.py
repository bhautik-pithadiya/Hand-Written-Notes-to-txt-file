from flask import Flask, request, jsonify
import numpy as np
from google.cloud import vision
from google.cloud.vision import types
import os
import io
import numpy as np
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from gensim.parsing.preprocessing import remove_stopwords
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
from nltk.tokenize import word_tokenize, sent_tokenize
import json

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'innate-empire-357206-1bd0f376b0cf.json'
client = vision.ImageAnnotatorClient()

tokenizer = AutoTokenizer.from_pretrained(
    'sentence-transformers/bert-base-nli-mean-tokens')
model = AutoModel.from_pretrained(
    'sentence-transformers/bert-base-nli-mean-tokens')


def img_to_txt(img):

    # pass the image with text(handwritten) to google vision output string
    with io.open(img, 'rb') as image_file:
        content = image_file.read()

    image_v = vision.types.Image(content=content)
    response = client.document_text_detection(image=image_v)
    docText = response.full_text_annotation.text
    # print(docText)
    return " ".join(docText.splitlines())


def process_vector(sent):

    # for googles Universal Sentence encoder
    #embeddings = model(sent)
    # return embeddings

    # for BERT with pre processing steps
    token = {'input_ids': [], 'attention_mask': []}
    for sentence in sent:
        # encode each sentence, append to dictionary
        new_token = tokenizer.encode_plus(sentence, max_length=128,
                                          truncation=True, padding='max_length',
                                          return_tensors='pt')
        token['input_ids'].append(new_token['input_ids'][0])
        token['attention_mask'].append(new_token['attention_mask'][0])
    # reformat list of tensors to single tensor
    token['input_ids'] = torch.stack(token['input_ids'])
    token['attention_mask'] = torch.stack(token['attention_mask'])
    output = model(**token)
    embeddings = output.last_hidden_state
    att_mask = token['attention_mask']
    mask = att_mask.unsqueeze(-1).expand(embeddings.size()).float()
    mask_embeddings = embeddings * mask
    summed = torch.sum(mask_embeddings, 1)
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / summed_mask
    mean_pooled_fin = mean_pooled.detach().numpy()
    return mean_pooled_fin

# for BERT model
    #embeddings = model.encode(sent, convert_to_tensor=True)
    #
    # return(embeddings)

# calculate percentage of similarity


def cal_sim(processed_correct_vector, processed_answer_vector):
    arr = []
    arr = cosine_similarity(
        processed_correct_vector[0:],
        processed_answer_vector[0:]
    )
    sim = list(map(max, arr))
    avg = np.sum(sim, dtype=np.float64)/len(sim)
    rounde_and_percentage = round(float(avg) * 100)

    return rounde_and_percentage


def marks(total_marks, rounde_and_percentage):
    marks_percentage = (total_marks*rounde_and_percentage)/100
    final_marks = round(float(marks_percentage))
    return final_marks

####
####GRADING BASED ON PROVIDED TAG WORDS STRING####
####


string.punctuation


def remove_punctuation(crr_list):
    punctuationfree = "".join(
        [i for i in crr_list if i not in string.punctuation])
    return punctuationfree.lower()


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def compare_with_tags(tags, lemma_tokens, total_marks):
    words_found = 0
    for word in tags:
        if word in lemma_tokens:
            words_found = words_found+1
    percentage = (words_found/len(tags))*100
    final_percentage = round(float(percentage))
    marks_percentage = (total_marks*final_percentage)/100
    final_marks = round(float(marks_percentage))
    return final_marks


def compare_and_grade(crr_str, detected_ans, tags, total_marks):

    correct_string = crr_str
    provided_ans_list = detected_ans

    processed_correct_vector = process_vector(correct_string)
    processed_answer_vector = process_vector(provided_ans_list)

    sim = cal_sim(processed_correct_vector, processed_answer_vector)
    if sim >= 100:
        sim = 100
    else:
        sim = sim

    gen_marks = marks(total_marks, sim)

    pc_free = remove_punctuation(provided_ans_list)
    filtered_sentence = remove_stopwords(pc_free)

    lemmatizer = WordNetLemmatizer()
    lemma_tokens = [lemmatizer.lemmatize(w, get_wordnet_pos(
        w)) for w in nltk.word_tokenize(filtered_sentence)]

    final_score = compare_with_tags(tags, lemma_tokens, total_marks)
    if gen_marks >= final_score:
        return gen_marks
    else:
        return final_score


# def transform_image(pillow_image):
#     data = np.asarray(pillow_image)
#     data = data / 255.0
#     data = data[np.newaxis, ..., np.newaxis]
#     # --> [1, x, y, 1]
#     data = tf.image.resize(data, [28, 28])
#     return data


# def predict(x):
#     predictions = model(x)
#     predictions = tf.nn.softmax(predictions)
#     pred0 = predictions[0]
#     label0 = np.argmax(pred0)
#     return label0


app = Flask(__name__)


# api = Api(app)
# parser = reqparse.RequestParser()
# parser.add_argument('student_answer', type=string)
# parser.add_argument('correct_answer', type=string)
# parser.add_argument('tags', type=list)
# parser.add_argument('max_marks', type=int)


# @app.route("/", methods=["GET", "POST"])


@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # ABC = parser.parse_args()
        # file = request.files.get('img')
        # if file is None or file.filename == "":
        #     return jsonify({"error": "no file"})

        data = json.loads(request.data)
        correct_answer = data['correct_answer']
        student_answer = data['student_answer']
        tags = data['tags']
        max_marks = data['max_marks']
        # json = request.get_json()
        # correct_answer = json["correct_answer"]
        # tags = json["tags"]
        # max_marks = json["max_marks"]
        try:
            # answer = file.read()
            # student_answer = img_to_txt(answer)
            # image_bytes = file.read()
            # pillow_img = Image.open(io.BytesIO(image_bytes)).convert('L')
            # student_answer = img_to_txt(pillow_img)
            score = compare_and_grade(
                correct_answer, student_answer, tags, max_marks)
            sample = {"score": score}
            return jsonify(sample)
        except Exception as e:
            return jsonify({"error": str(e)})

    return "OK"


if __name__ == "__main__":
    app.run(debug=True)
