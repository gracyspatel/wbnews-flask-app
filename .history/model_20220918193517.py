import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

df = pd.read_csv('bbc-text.csv')
df.head()

#News Categories
pd.unique(df['category'])

sns.countplot(df.category)

#Tokenization
TOKENIZED_WORDS = []

for word in df['text']:
    TOKENIZED_WORDS.append(word_tokenize(word.lower()))

#Text is now tokenized
for words in TOKENIZED_WORDS[0:1]:
    print(words)

#Pickling TOKENIZED_WORDS
file = "pickle/TOKENIZED_WORDS.pkl"
fileobj = open(file, 'wb')
pickle.dump(TOKENIZED_WORDS, fileobj)
fileobj.close()

#Removing stopwords and punctuation
stop_words = set(stopwords.words("english"))

punctuations = set(string.punctuation)

FILTERED_TEXT = []

for text in TOKENIZED_WORDS:
    temp_text = []
    for i in text:
        if((i not in stop_words) and (i not in punctuations) and (i != "'s'")):
            temp_text.append(i)
            
    FILTERED_TEXT.append(temp_text)
    
print("\nFiltered Text : ")
print(FILTERED_TEXT[0:1])

#Pickling FILTERED_TEXT
file = "pickle/FILTERED_TEXT.pkl"
fileobj = open(file, 'wb')
pickle.dump(FILTERED_TEXT, fileobj)
fileobj.close()

#Stemmin using Porter Stemmer
porter = PorterStemmer()

STEMMED_TEXT = []

for text in FILTERED_TEXT:
    temp_text = []
    for word in text:
        temp_text.append(porter.stem(word))
        
    STEMMED_TEXT.append(" ".join(temp_text))
    
print("Stemmed Text : ")
print(STEMMED_TEXT[0:2])

#Pickling STEMMED_TEXT
file = "pickle/STEMMED_TEXT.pkl"
fileobj = open(file, 'wb')
pickle.dump(STEMMED_TEXT, fileobj)
fileobj.close()

df.head()

#Replacing text with STEMMED_TEXT
df = df.drop(['text'], axis=1)
df.insert(0, "text", STEMMED_TEXT, True)

df.head()

#Encoding News Category
labelencoder = LabelEncoder()

df.insert(2, "encoded_category", labelencoder.fit_transform(df['category']), True)

df.head()

#Business = 0
#Entertainment = 1
#Politics = 2
#Sport = 3
#Tech = 4

#Pickling Dataframe
file = "pickle/DATAFRAME.pkl"
fileobj = open(file, 'wb')
pickle.dump(df, fileobj)
fileobj.close()

# Naive Bayes Classification
file = "pickle/DATAFRAME.pkl"
fileobj = open(file, 'rb')
df = pickle.load(fileobj)
fileobj.close()

print(type(df))
df.head()

#News text
X = df['text']

#Encoded News Category
y = df['encoded_category']

#Splitting the data set into Training and Testing set

#Testing_set = 25% and Training_set = 75%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 51)

print("Shape of X : " + str(X.shape))
print("Shape of y : " + str(y.shape))

print("\nShape of X_train : " + str(X_train.shape))
print("Shape of y_train : " + str(y_train.shape))
print("Shape of X_test : " + str(X_test.shape))
print("Shape of y_test : " + str(X_test.shape))

# Feature Selection : TF-IDF Approach
# Term Frequency - Inverse Document Frequency

#Feature Extraction

#Instantiating TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

#Fitting and Transforming Taining Data(X_train)
tfidf_X_train = tfidf_vectorizer.fit_transform(X_train.values)

#Tramsforming Testing Data(X_test)
tfidf_X_test = tfidf_vectorizer.transform(X_test.values)

#Saving tfidf_vectorizer
pickle.dump(tfidf_vectorizer, open("pickle/tfidf_vectorizer.pkl", 'wb'))

#Multimomial Naive Bayes Classifier

#Instantiating Naive Bayes Classifier with alpha = 1.0
nb_classifier = MultinomialNB()

#Fitting nb_classifier to Training Data
nb_classifier.fit(tfidf_X_train, y_train)

#Saving nb_classifier for tfidf_vectorizer
pickle.dump(nb_classifier, open("pickle/nb_classifier_for_tfidf_vectorizer.pkl", 'wb'))

pred = nb_classifier.predict(tfidf_X_test)

#Accuracy Score and Confusion Matrix
print("Multinomial Naive Bayes : (TF-IDF Approach) \n")

#Accuracy
a_score = metrics.accuracy_score(y_test, pred)
print("Accuracy : " + str("{:.2f}".format(a_score*100)), '%\n')

#Confusion MAtrix
confusion_matrix = metrics.confusion_matrix(y_test, pred)

print("Confusion Matrix : ")
print(confusion_matrix)

#Laplace Smooting (Tunning parameter - alpha)

alphas = np.arange(0,1,0.1)

#Function for traing nb_classifier with differnt alpha values
def train_predict(alpha):
    #Instantiating Naive Bayes Classifier
    nb_classifier = MultinomialNB(alpha=alpha)
    
    #Fitting nb_classifier to traning data
    nb_classifier.fit(tfidf_X_train, y_train)
    
    #Prediction
    pred = nb_classifier.predict(tfidf_X_test)
    
    #Accuracy Score
    a_score = metrics.accuracy_score(y_test, pred)
    
    return a_score

for alpha in alphas:
    print("Alpha : ", alpha)
    print("Accuracy Score : ", train_predict(alpha))
    print()


#Prediction of New Category
tfidf_vectorizer = pickle.load(open("pickle/tfidf_vectorizer.pkl", 'rb'))
nb_classifier = pickle.load(open("pickle/nb_classifier_for_tfidf_vectorizer.pkl", 'rb'))

#Values encoded by LabelEncoder
encoded = {0:'Business', 1:'Entertainment', 2:'Politics', 3:'Sports', 4:'Technology'}

#Input
user_text = [input("Enter the news : ")]

#Transformation and Prediction of user text
count = tfidf_vectorizer.transform(user_text)
prediction = nb_classifier.predict(count)

print("\nNews Category : ", encoded[prediction[0]])