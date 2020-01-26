import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import classification_report

np.random.seed(500)

Dataset = pd.read_csv("test_final.csv", encoding='latin-1')
Dataset['text'].dropna(inplace=True)
Dataset['text'] = [entry.lower() for entry in Dataset['text']]
Dataset['text'] = [word_tokenize(entry) for entry in Dataset['text']]

tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index, entry in enumerate(Dataset['text']):
    Final_words = []

    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = WordNetLemmatizer().lemmatize(word, tag_map[tag[0]])
            Final_words.append(word_Final)
    Dataset.loc[index, 'text_final'] = str(Final_words)

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Dataset['text_final'],
                                                                    Dataset['label'],
                                                                    test_size=0.2)

Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

Tfidf_vect = TfidfVectorizer(max_features=8000)
Tfidf_vect.fit(Dataset['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

print(Tfidf_vect.vocabulary_)
print(Train_X_Tfidf)

# Classifier - Algorithm - Naive Bayes
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf, Train_Y)
predictions_NB = Naive.predict(Test_X_Tfidf)
print("Naive Bayes Classification Report: ")
print(classification_report(Test_Y, predictions_NB))

# Classifier - Algorithm - SVM
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf, Train_Y)
predictions_SVM = SVM.predict(Test_X_Tfidf)
print("SVM Classification Report: ")
print(classification_report(Test_Y, predictions_SVM))
