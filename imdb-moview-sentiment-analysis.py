# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import requests   # Importing requests to extract content from a url
from bs4 import BeautifulSoup as bs # Beautifulsoup is for web scrapping...used to scrap specific content 
import re 
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# %%
# creating empty reviews list 
movie_reviews=[]


# %%
for i in range(1,21):
  ip=[]  
  url="https://www.imdb.com/title/tt5433138/reviews?ref_=tt_urv"+str(i)
  response = requests.get(url)
  soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content 
  reviews = soup.find_all("div", attrs={"class": "text show-more__control"})# Extracting the content under specific tags  
  for i in range(len(reviews)):
    ip.append(reviews[i].text)  

    movie_reviews= movie_reviews+ip  # adding the reviews of one page to empty list which in future contains all the reviews


# %%
movie_reviews


# %%
# writng reviews in a text file 
with open("movie-reviews.txt","w",encoding='utf8') as output:
    output.write(str(movie_reviews))


# %%
# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(movie_reviews)


# %%
import nltk
# from nltk.corpus import stopwords|


# %%
# Removing unwanted symbols incase if exists
ip_rev_string = re.sub("[^A-Za-z" "]+"," ", ip_rev_string).lower()
ip_rev_string = re.sub("[0-9" "]+"," ", ip_rev_string)


# %%
# words that contained in iphone XR reviews
ip_reviews_words = ip_rev_string.split(" ")
ip_reviews_words


# %%
#TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ip_reviews_words, use_idf=True,ngram_range=(1, 3))
X = vectorizer.fit_transform(ip_reviews_words)


# %%
with open("stop.txt","r") as sw:
    stop_words = sw.read()


# %%
stop_words = stop_words.split("\n")

stop_words.extend(["imdb","laptop","time","windows","laptop","device","screen","battery","product","good","day","price"])


# %%
ip_reviews_words = [w for w in ip_reviews_words if not w in stop_words]

# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(ip_reviews_words)
wordcloud_ip = WordCloud(
                      background_color='White',
                      width=1800,
                      height=1400
                     ).generate(ip_rev_string)

plt.imshow(wordcloud_ip)
plt.title("Main Keywords that are extracted from the reviews of Fast and Furios 9 movie")
plt.show()


# %%
# WordCloud can be performed on the string inputs.
# Corpus level word cloud
# positive words # Choose the path for +ve words stored in system
with open(r"positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")


# %%
# Positive word cloud
# Choosing the only words which are present in positive words
ip_pos_in_pos = " ".join ([w for w in ip_reviews_words if w in poswords])

wordcloud_pos_in_pos = WordCloud(
                      background_color='White',
                      width=1800,
                      height=1400
                     ).generate(ip_pos_in_pos)
plt.figure(2)
plt.title("Positive words Keywords that are extracted from the reviews of Fast and Furios 9 movie")
plt.imshow(wordcloud_pos_in_pos)
plt.show()


# %%
# negative words Choose path for -ve words stored in system
with open(r"negative-words.txt", "r") as neg:
  negwords = neg.read().split("\n")

# negative word cloud
# Choosing the only words which are present in negwords
ip_neg_in_neg = " ".join ([w for w in ip_reviews_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_neg_in_neg)
plt.figure(3)
plt.title("Negative  Keywords that are extracted from the reviews of Fast and Furios 9 movie")
plt.imshow(wordcloud_neg_in_neg)
plt.show()

# %% [markdown]
# # **Word cloud with Bigram**

# %%
# wordcloud with bigram
nltk.download('punkt')
from wordcloud import WordCloud, STOPWORDS


# %%
WNL = nltk.WordNetLemmatizer() # it groups up similar word i.e words with similar meaning

# Lowercase and tokenize
text = ip_rev_string.lower()

# Remove single quote early since it causes problems with the tokenizer.
text = text.replace("'", "")


# %%
tokens = nltk.word_tokenize(text)
text1 = nltk.Text(tokens)

# Remove extra chars and remove stop words.
text_content = [''.join(re.split("[ .,;:!?‘’``''@#$%^_&*()<>{}~\n\t\\\-]", word)) for word in text1]
text_content


# %%
# Create a set of stopwords
stopwords_wc = set(STOPWORDS)
customised_words = ['price', 'great'] # If you want to remove any particular word form text which does not contribute much in meaning


# %%
new_stopwords = stopwords_wc.union(customised_words)
# Remove stop words
text_content = [word for word in text_content if word not in new_stopwords]

# Take only non-empty entries
text_content = [s for s in text_content if len(s) != 0]


# %%
# Best to get the lemmas of each word to reduce the number of similar words
text_content = [WNL.lemmatize(t) for t in text_content]


# %%
nltk_tokens = nltk.word_tokenize(text)  
bigrams_list = list(nltk.bigrams(text_content))
print(bigrams_list)


# %%
dictionary2 = [' '.join(tup) for tup in bigrams_list]
print (dictionary2)


# %%
# Using count vectoriser to view the frequency of bigrams
from sklearn.feature_extraction.text import CountVectorizer


# %%
vectorizer = CountVectorizer(ngram_range=(2, 2))
bag_of_words = vectorizer.fit_transform(dictionary2)
vectorizer.vocabulary_


# %%
sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
print(words_freq[:100])


# %%
# Generating wordcloud
words_dict = dict(words_freq)
WC_height = 2000
WC_width = 2500
WC_max_words = 200
wordCloud = WordCloud(background_color='white',max_words=WC_max_words, height=WC_height, width=WC_width, stopwords=new_stopwords)
wordCloud.generate_from_frequencies(words_dict)

plt.figure(figsize= (70,100))
plt.figure(4)
plt.title('Most frequently occurring bigrams connected by same colour and font size')
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# %%
import pandas as pd
df = pd.DataFrame(words_freq, columns = ['Words', 'Ocuurence'])
df


# %%
from nltk.corpus import stopwords


# %%
# Lemmatization
# Lemmatization looks into dictionary words
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer


# %%
lemmatizer = WordNetLemmatizer()
for i in df["Words"]:
    lemmatizer.lemmatize(i)
    


# %%
#Chunking (Shallow Parsing) - Identifying named entities
nltk.download('maxent_ne_chunker')
nltk.download('words')


# %%
# Number of words
from textblob import TextBlob


# %%
# Detect presence of wh words
wh_words = set(['why', 'who', 'which', 'what', 'where', 'when', 'how'])
df['is_wh_words_present'] = df['Words'].apply(lambda x : True if len(set(TextBlob(str(x)).words).intersection(wh_words))>0 else False)
df['is_wh_words_present']


# %%
# Polarity
df['polarity'] = df['Words'].apply(lambda x : TextBlob(str(x)).sentiment.polarity)
df['polarity']


# %%
# Subjectivity
df['subjectivity'] = df['Words'].apply(lambda x : TextBlob(str(x)).sentiment.subjectivity)
df['subjectivity']


# %%
df.head(5)


# %%
# assign reviews with score > 3 as positive sentiment
# score < 3 negative sentiment
# remove score = 3
df = df[df['polarity'] != 0]
df['sentiment'] = df['polarity'].apply(lambda rating : +1 if rating > 0 else -1)


# %%
df.head(10)


# %%
# split df - positive and negative sentiment:
positive = df[df['sentiment'] == 1]
negative = df[df['sentiment'] == -1]


# %%
import plotly.express as px


# %%
df['sentimentt'] = df['sentiment'].replace({-1 : 'negative'})
df['sentimentt'] = df['sentimentt'].replace({1 : 'positive'})
fig = px.histogram(df, x="sentimentt")
fig.update_traces(marker_color="indianred",marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5)
fig.update_layout(title_text=' Fast and Furios 9 Movie reviews Sentiment Analysis')
fig.show()


# %%
df


# %%
dfNew = df[['Words','sentiment']]
dfNew.head()


# %%
import numpy as np


# %%
# random split train and test data
index = df.index
df['random_number'] = np.random.randn(len(index))
train = df[df['random_number'] <= 0.8]
test = df[df['random_number'] > 0.8]


# %%
from sklearn.feature_extraction.text import CountVectorizer


# %%
# count vectorizer:
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform(train['Words'])
test_matrix = vectorizer.transform(test['Words'])


# %%
# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


# %%
X_train = train_matrix
X_test = test_matrix
y_train = train['sentiment']
y_test = test['sentiment']


# %%
lr.fit(X_train,y_train)


# %%
predictions = lr.predict(X_test)
predictions


# %%
# find accuracy, precision, recall:
from sklearn.metrics import confusion_matrix,classification_report
new = np.asarray(y_test)
confusion_matrix(predictions,y_test)


# %%
print(classification_report(predictions,y_test))

# %% [markdown]
# # **Extra Part**

# %%
corpus = [ip_rev_string]

bag_of_words_model = CountVectorizer()
print(bag_of_words_model.fit_transform(corpus).todense()) # bag of words


# %%
bag_of_word_df = pd.DataFrame(bag_of_words_model.fit_transform(corpus).todense())
bag_of_word_df.columns = sorted(bag_of_words_model.vocabulary_)
bag_of_word_df.head()


# %%
# Bag of word model for top 5 frequent terms
bag_of_words_model_small = CountVectorizer(max_features=5)
bag_of_word_df_small = pd.DataFrame(bag_of_words_model_small.fit_transform(corpus).todense())
bag_of_word_df_small.columns = sorted(bag_of_words_model_small.vocabulary_)
bag_of_word_df_small.head()


# %%
# TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer


# %%
tfidf_model = TfidfVectorizer()
print(tfidf_model.fit_transform(corpus).todense())


# %%
tfidf_df = pd.DataFrame(tfidf_model.fit_transform(corpus).todense())
tfidf_df.columns = sorted(tfidf_model.vocabulary_)
tfidf_df.head()


# %%
# TFIDF for top 5 frequent terms
tfidf_model_small = TfidfVectorizer(max_features=5)
tfidf_df_small = pd.DataFrame(tfidf_model_small.fit_transform(corpus).todense())
tfidf_df_small.columns = sorted(tfidf_model_small.vocabulary_)
tfidf_df_small.head()


