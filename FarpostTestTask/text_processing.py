import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter

def load_dataset(url):
    dataset = []
    with open(url, 'r', encoding='utf-8') as file:
        for line in file:
            dataset.append(line.strip())
    return dataset

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('russian'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def analyze_dataset(dataset):
    all_words = []
    for text in dataset:
        tokens = preprocess_text(text)
        all_words.extend(tokens)
    word_freq = Counter(all_words)
    return word_freq.most_common(10)

def visualize_word_frequency(common_words):
    for word, freq in common_words:
        print(f"{word}: {freq}")