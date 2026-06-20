import random
import json
import pickle
import nltk
import numpy as np

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

# Try to load intents from expected location; fall back to AI-Chatbot/data if missing
try:
    intents = json.loads(open('data/intents.json').read())
except Exception:
    try:
        intents = json.loads(open('AI-Chatbot/data/interns.json').read())
    except Exception:
        intents = {"intents": []}

# Try to load trained resources; if any are missing, fall back to a simple rule-based predictor
use_model = True
try:
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    model = load_model('model/chatbot_model.h5')
except Exception:
    use_model = False
    words = []
    classes = []

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [
        lemmatizer.lemmatize(word.lower())
        for word in sentence_words
    ]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1

    return np.array(bag)

def predict_class(sentence):
    if use_model:
        bow = bag_of_words(sentence)

        result = model.predict(
            np.array([bow]),
            verbose=0
        )[0]

        tag_index = np.argmax(result)

        return classes[tag_index]
    # Simple fallback: match lemmatized tokens against patterns in intents
    sentence_words = clean_up_sentence(sentence)
    best_tag = None
    best_count = 0
    for intent in intents.get('intents', []):
        count = 0
        for pattern in intent.get('patterns', []):
            # tokenize pattern and compare words
            pattern_words = [lemmatizer.lemmatize(w.lower()) for w in nltk.word_tokenize(pattern)]
            for w in sentence_words:
                if w in pattern_words:
                    count += 1
        if count > best_count:
            best_count = count
            best_tag = intent.get('tag')
    if best_tag:
        return best_tag
    # no match found: return first intent tag if available
    intents_list = intents.get('intents', [])
    if intents_list:
        return intents_list[0].get('tag')
    return None

def get_response(tag):
    if not tag:
        return "Sorry, I don't understand."

    for intent in intents.get('intents', []):
        if intent.get('tag') == tag:
            responses = intent.get('responses', [])
            if responses:
                return random.choice(responses)
    return "Sorry, I don't understand."

while True:
    message = input("You: ")

    if message.lower() == "quit":
        break

    tag = predict_class(message)

    response = get_response(tag)

    print("Bot:", response)
    