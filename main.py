import numpy as np
import pandas as pd
from collections import Counter

df = pd.read_csv('Reviews.csv')
#print(df.info())

#Print most commonly used words
#mport nltk
#from nltk.corpus import stopwords
#from wordcloud import WordCloud,STOPWORDS
#stopwords = set(STOPWORDS)
#stopwords.update(["br", "href"])
#textt = " ".join(review for review in df.Text)
#wordcloud = WordCloud(stopwords=stopwords).generate(textt)
#print(wordcloud)
#frequent_words = Counter(" ".join(df["Text"]).split()).most_common(100)
#print(Counter(" ".join(df["Text"]).split()).most_common(100))

# assign reviews with score > 3 as positive sentiment
# score < 3 negative sentiment and remove score = 3
positive = df[df['Score'] > 3]
negative = df[df['Score'] < 3]

df = df[df['Score'] != 3]
df['sentiment'] = df['Score'].apply(lambda rating : +1 if rating > 3 else -1)

#print(positive.info())
#print(negative.info())

#positive_frequent_words = Counter(" ".join(positive["Text"]).split()).most_common(100)
#negative_frequent_words = Counter(" ".join(negative["Text"]).split()).most_common(100)

#print(positive_frequent_words)
#print(negative_frequent_words)

def remove_punctuation(text):
    final = "".join(u for u in text if u not in ("?", ".", ";", ":",  "!",'"'))
    return final

df['Text'] = df['Text'].apply(remove_punctuation)
df = df.dropna(subset=['Summary'])
df['Summary'] = df['Summary'].apply(remove_punctuation)

dfNew = df[['Summary','sentiment']]
#print(dfNew.info())

# random split train and test data
index = df.index
df['random_number'] = np.random.randn(len(index))
train = df[df['random_number'] <= 0.8]
test = df[df['random_number'] > 0.8]

# count vectorizer:
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform(train['Summary'])
test_matrix = vectorizer.transform(test['Summary'])

# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='lbfgs', max_iter=1000)

X_train = train_matrix
X_test = test_matrix
y_train = train['sentiment']
y_test = test['sentiment']

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report
new = np.asarray(y_test)
#cm = confusion_matrix(y_pred,y_test)
#print(cm)

#print(classification_report(y_pred,y_test))

#My unsuccessful attempt to make this code interactive
import csv
with open('input.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Summary','sentiment'])
    with open('input.txt') as f:
        for line in f:
            #print(line)
            writer.writerow([line, 1])


inp = pd.read_csv('input.csv')
#print(inp.info())
test_matrix = vectorizer.transform(inp['Summary'])
res = lr.predict(test_matrix)
print('Predicted results for given arguement: ', res)