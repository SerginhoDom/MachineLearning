import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from Levenshtein import distance as levenshtein_distance

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('russian'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def correct_errors(input_text, word_freq):
    input_tokens = preprocess_text(input_text)
    corrected_text = []
    for token in input_tokens:
        if token not in word_freq:
            # Если слово отсутствует в частотном словаре,
            # заменяем его на наиболее близкое слово по расстоянию Левенштейна
            closest_word = min(word_freq, key=lambda x: levenshtein_distance(x, token))
            corrected_text.append(closest_word)
        else:
            corrected_text.append(token)
    return ' '.join(corrected_text)