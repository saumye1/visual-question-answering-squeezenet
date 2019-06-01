import json
from scipy.misc import imread, imresize
import os

def load_answers(path=''):
    with open(path or './answers_val.json') as f:
        data = json.load(f)
        answers = data['annotations']
        return { answer['question_id']: answer['multiple_choice_answer'] for answer in answers }

def load_questions(path=''):
    with open(path or './questions_val.json') as f:
        data = json.load(f)
        questions = data['questions']
        return questions

def get_distinct_words(questions):
    questions_words = [ "".join(question['question'].lower().split("?")) for question in questions]
    words = [set(q.split(" ")) for q in questions_words]
    set_words = set()
    for s in words:
        set_words = set_words.union(s)
    
    return set_words

def load_data(path=''):
    questions = load_questions(path)
    answers = load_answers(path)
    data = [ 
        {
                "question" : q['question'].split("?")[0].lower(),
                "question_id" : q['question_id'],
                "image_id" : q['image_id'],
                "answer" : answers[q['question_id']].lower()
        } for q in questions]
    return data

def load_images(image_ids, path='', folder_name = 'val2014'):
    image_names = os.listdir(path + './' + folder_name)
    images_filtered = []
    for im_needed in image_ids:
        im_str = '0' * (12 - im_needed.__str__().__len__()) + im_needed.__str__() + '.jpg'
        for im in image_names:
            if im.endswith(im_str):
                images_filtered.append(im)

    images = [imread(path + './' + folder_name + '/' + im_name) for im_name in images_filtered]
    return images


