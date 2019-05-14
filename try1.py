import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import re
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import f1_score, accuracy_score
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample.csv')


stop_words = set(stopwords.words('english'))
stop = [x.lower() for x in stop_words]
lemma = WordNetLemmatizer()


def clean(text):
    text = text.lower()
    # keep alphanumeric characters only
    text = re.sub('\W+', ' ', text).strip()
    text = text.replace('user', '')
    # tokenize
    text_token = word_tokenize(text)
    # replace shortcuts using dict
    full_words = []
    for token in text_token:
        if token in shortcuts.keys():
            token = shortcuts[token]
        full_words.append(token)
#     text = " ".join(full_words)
#     text_token = word_tokenize(text)
    # stopwords removal
#     words = [word for word in full_words if word not in stop]
    words_alpha = [re.sub(r'\d+', '', word) for word in full_words]
    words_big = [word for word in words_alpha if len(word)>2]
    stemmed_words = [lemma.lemmatize(word) for word in words_big]
    # join list elements to string
    clean_text = " ".join(stemmed_words)
    clean_text = clean_text.replace('   ', ' ')
    clean_text = clean_text.replace('  ', ' ')
    return clean_text

hypocrite = []
for i in range(len(train['tweet'])):
    if 'hypocrite' in train['tweet'][i]:
        if train['label'][i] == 1:
            hypocrite.append('racist')
        else:
            hypocrite.append('good')
    else:
        hypocrite.append('good')
df = pd.DataFrame(columns=['hypocrite'], data=hypocrite)
print(df['hypocrite'].value_counts())

train['hypocrite'] = hypocrite

train['combined'] = train['tweet'].apply(str) + ' ' + train['hypocrite'].apply(str)

X_train = train.combined
y = train.label
X_test = test.tweet



vectorizer = CountVectorizer(max_df=0.5)
# vectorizer = TfidfVectorizer(ngram_range=(1,3), max_df=0.5)

X = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

model = LinearSVC(penalty='l2', C=0.5, dual=False, random_state=0, max_iter=1000)
print(model)

# split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=0)

# calculate f1 score
model.fit(X_train,y_train)
y_pred = model.predict(X_val)
print('Accuracy:', accuracy_score(y_pred, y_val))
print("F1 Score: ", f1_score(y_pred, y_val))


df = pd.DataFrame()
df['y_pred'] = y_pred



# train model with full data and predict for new samples
model.fit(X, y)
y_pred = model.predict(X_test)


df = pd.DataFrame()
df['y_pred'] = y_pred


submission['label'] = y_pred
submission.to_csv('sample3.csv', index=False)



