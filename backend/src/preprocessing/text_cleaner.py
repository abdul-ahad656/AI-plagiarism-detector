import re
import nltk
import spacy
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    # Remove special characters and digits
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.lower()
    
    # Tokenize and remove stopwords
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize using spaCy
    doc = nlp(" ".join(tokens))
    lemmatized = [token.lemma_ for token in doc]

    return " ".join(lemmatized)
