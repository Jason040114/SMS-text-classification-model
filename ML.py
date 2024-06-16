import pandas as pd
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
import nltk
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
# 加载数据集，注意sep参数表⽰更换分隔符
dataset = pd.read_csv("sms.csv")
dataset['sms'] = dataset['sms'].astype(str)
# 前5000句作为训练集，5001到末尾作为验证集
train_dataset = dataset.head(5000)
valid_dataset = dataset.tail(-5000)
def ML():
    def tokenizer(text):
        text=text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        t = text.split()
        t=[i for i in t if i.isalpha()]
        t=[WordNetLemmatizer().lemmatize(i) for i in t]
        return t
    vec = TfidfVectorizer(stop_words="english", min_df=0, max_df=1.0, ngram_range=(1, 2),tokenizer=tokenizer)
    train = vec.fit_transform(train_dataset.sms)
    valid=vec.transform(valid_dataset.sms)
    model = MultinomialNB(alpha=0.1)
    model.fit(train, train_dataset['label'])
    predicted_labels = model.predict(valid)
    score = f1_score(valid_dataset.label, predicted_labels)
    print(score)
