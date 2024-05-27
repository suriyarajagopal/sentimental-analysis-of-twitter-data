import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import gensim
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Loading data
train = pd.read_csv('/content/train_tweet.csv')
test = pd.read_csv('/content/test_tweets.csv')

# Checking for Null Values
print(train.isnull().any())
print(test.isnull().any())

# EDA
train['label'].value_counts().plot.bar(color='pink', figsize=(6, 4))
plt.title('Distribution of Tweets')
plt.show()

train['len'] = train['tweet'].str.len()
test['len'] = test['tweet'].str.len()

train.head()

train.groupby('label').describe()
train.groupby('len')['label'].mean().plot.hist(color='black', figsize=(6, 4))

# train.groupby('len').mean()['label'].plot.hist(color='black', figsize=(6, 4))
plt.title('Variation of Length')
plt.xlabel('Length')
plt.show()

# Word Frequency Analysis
cv = CountVectorizer(stop_words='english')
words = cv.fit_transform(train.tweet)
sum_words = words.sum(axis=0)
words_freq = [(word, sum_words[0, i]) for word, i in cv.vocabulary_.items()]
words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])
frequency.head(30).plot(x='word', y='freq', kind='bar', figsize=(15, 7), color='blue')
plt.title("Most Frequently Occurring Words - Top 30")
plt.show()

wordcloud = WordCloud(background_color='white', width=1000, height=1000).generate_from_frequencies(dict(words_freq))
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud)
plt.title("WordCloud - Vocabulary from Reviews", fontsize=22)
plt.show()

# Hashtag Analysis
def hashtag_extract(x):
    hashtags = []
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
    return hashtags

HT_regular = hashtag_extract(train['tweet'][train['label'] == 0])
HT_negative = hashtag_extract(train['tweet'][train['label'] == 1])

HT_regular = sum(HT_regular, [])
HT_negative = sum(HT_negative, [])

a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()), 'Count': list(a.values())})
d = d.nlargest(columns="Count", n=20)
plt.figure(figsize=(16, 5))
ax = sns.barplot(data=d, x="Hashtag", y="Count")
ax.set(ylabel='Count')
plt.title('Top 20 Hashtags in Non-Racist/Sexist Tweets')
plt.show()

a = nltk.FreqDist(HT_negative)
d = pd.DataFrame({'Hashtag': list(a.keys()), 'Count': list(a.values())})
d = d.nlargest(columns="Count", n=20)
plt.figure(figsize=(16, 5))
ax = sns.barplot(data=d, x="Hashtag", y="Count")
ax.set(ylabel='Count')
plt.title('Top 20 Hashtags in Racist/Sexist Tweets')
plt.show()

# Word Embedding using Word2Vec
tokenized_tweet = train['tweet'].apply(lambda x: x.split())
model_w2v = gensim.models.Word2Vec(tokenized_tweet, vector_size=200, window=5, min_count=2, sg=1, hs=0, negative=10, workers=2, seed=34)

# model_w2v = gensim.models.Word2Vec(tokenized_tweet, size=200, window=5, min_count=2, sg=1, hs=0, negative=10, workers=2, seed=34)
model_w2v.train(tokenized_tweet, total_examples=len(train['tweet']), epochs=20)

# Data Preprocessing
nltk.download('stopwords')
train_corpus = []
for i in range(len(train)):
    review = re.sub('[^a-zA-Z]', ' ', train['tweet'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    train_corpus.append(review)

test_corpus = []
for i in range(len(test)):
    review = re.sub('[^a-zA-Z]', ' ', test['tweet'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    test_corpus.append(review)

cv = CountVectorizer(max_features=2500)
x_train = cv.fit_transform(train_corpus).toarray()
y_train = train.iloc[:, 1].values
x_test = cv.transform(test_corpus).toarray()

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_valid = sc.transform(x_valid)
x_test = sc.transform(x_test)

# Modeling
model_rf = RandomForestClassifier()
model_rf.fit(x_train, y_train)
y_pred_rf = model_rf.predict(x_valid)

print("Random Forest - Validation Accuracy:", model_rf.score(x_valid, y_valid))
print("Random Forest - F1 Score:", f1_score(y_valid, y_pred_rf))

model_lr = LogisticRegression()
model_lr.fit(x_train, y_train)
y_pred_lr = model_lr.predict(x_valid)

print("Logistic Regression - Validation Accuracy:", model_lr.score(x_valid, y_valid))
print("Logistic Regression - F1 Score:", f1_score(y_valid, y_pred_lr))

model_dt = DecisionTreeClassifier()
model_dt.fit(x_train, y_train)
y_pred_dt = model_dt.predict(x_valid)

print("Decision Tree - Validation Accuracy:", model_dt.score(x_valid, y_valid))
print("Decision Tree - F1 Score:", f1_score(y_valid, y_pred_dt))

model_svc = SVC()
model_svc.fit(x_train, y_train)
y_pred_svc = model_svc.predict(x_valid)

print("SVM - Validation Accuracy:", model_svc.score(x_valid, y_valid))
print("SVM - F1 Score:", f1_score(y_valid, y_pred_svc))

model_xgb = XGBClassifier()
model_xgb.fit(x_train, y_train)
y_pred_xgb = model_xgb.predict(x_valid)

print("XGBoost - Validation Accuracy:", model_xgb.score(x_valid, y_valid))
print("XGBoost - F1 Score:", f1_score(y_valid, y_pred_xgb))
