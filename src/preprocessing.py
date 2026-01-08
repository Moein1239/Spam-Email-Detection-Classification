import re
import string
import nltk
from nltk.corpus import stopwords

# برای اولین بار:
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    # lowercase
    text = text.lower()
    # حذف لینک
    text = re.sub(r'http\S+|www\S+', '', text)
    # حذف ایموجی و کاراکترهای خاص
    text = text.encode('ascii', 'ignore').decode()
    # حذف punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # حذف عدد
    text = re.sub(r'\d+', '', text)
    # حذف space اضافه
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_stopwords(text):
    tokens = text.split()
    filtered = [word for word in tokens if word not in stop_words]
    return " ".join(filtered)


def preprocess_text(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    return text
