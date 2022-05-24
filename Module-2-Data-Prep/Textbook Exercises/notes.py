'''
    this is a module that generates a series of transformation functions for text.
    BAG of WORDS analysis:
        pros:
            great for categorizing a body of text (politics, etc)
        cons:
            terrible for sentiment analysis.
    
    CONTENTS
    0. creating a pipeline
        -removing stop words and tokenizing
    1. creating a word counter 
    2. plotting and word clouds
    3. Inverse document frequency - term frequency (TF-IDF)
    4. Words-in-context analysis.
'''

#import the UN GA Debate
import pandas as pd
import numpy as np
import nltk
import re 
df = pd.read_csv('ungd.csv.gz')

def tokenize(text):
    return re.findall(r'[\w-]*\p{L}[\w-]*', text)

#error handle not having the stop words by downloading and trying again.

try:
    stopwords = set(nltk.corpus.stopwords.words('english'))
except:
    nltk.download('stopwords')
    stopwords = set(nltk.corpus.stopwords.words('english'))

# define a function that removes any tokens that are part of the stopwords list.
def remove_stop(tokens):
    return [t for t in tokens if t.lower() not in stopwords]

# define a function that transforms all the text in our dataset.
pipeline = [str.lower, tokenize, remove_stop]
def prepare(text, pipeline):
    tokens = text
    for transform in pipeline:
        tokens = transform(tokens)
    return tokens

# we can then apply this pipeline to each element of a dataframe to tokenize it.
df['tokens'] = df['text'].apply(prepare,pipeline=pipeline)

# we can also get the number of tokens for each row.
df['num_tokens'] = df['tokens'].map(len)

''' 
INTRODUCING THE NATIVE PYTHON COUNTER
this is a library that counts things. we can pass a list of tokens to it to get a count.
the counter object can also be updated, which means we can map it across a df to get a word count for the entire
dataset.
'''

from collections import Counter

#initial demonstration
tokens = prepare("She Likes my cats and my cats like my sofa.",pipeline)
counter = Counter(tokens)
print(counter)
print('-------------------------\n')

#you can also update the counter
more_tokens = tokenize("She likes dogs and cats.")
counter.update(more_tokens)
print(counter)
print('-------------------------\n')

#as well as map it across an entire dataframe.
#counter = Counter()
#df['tokens'].map(counter.update)

print(df.head(10))
#print('-------------------------\n')


'''
We can also create a data frame of word frequencies:
'''

def count_words(df, column='tokens', preprocess=None, min_freq=2):
    # process tokens and update counter
    def update(doc):
        tokens = doc if preprocess is None else preprocess(doc)
        counter.update(tokens)
    # create counter and run through all data
    counter = Counter()
    df[column].map(update)
    # transform counter into a DataFrame
    freq_df = pd.DataFrame.from_dict(counter, orient='index', columns=['freq'])
    freq_df = freq_df.query('freq >= @min_freq')
    freq_df.index.name = 'token'
    return freq_df.sort_values('freq', ascending=False)

freq = count_words(df)
print(freq.head(5))


"""
    CREATING A FREQUENCY DIAGRAM 
    we can use the built in pandas plot functionality rather than relying on matplotlib
"""
#we still need plt to expose it.
import matplotlib.pyplot as plt
ax = freq.head(15).plot(kind='barh', width=0.95)
ax.invert_yaxis()
ax.set(xlabel='Frequency', ylabel='Token', title='Top Words')
plt.show()


"""
    CREATING A WORD CLOUD
"""

from wordcloud import WordCloud
text = df.query("year==2015 and country=='USA'")['text'].values[0]
wc = WordCloud(max_words=100, stopwords=stopwords)
wc.generate(text)
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()

def wordcloud(word_freq, title=None, max_words=200, stopwords=None):
    wc = WordCloud(width=800, height=400,
    background_color= "black", colormap="Paired",
    max_font_size=150, max_words=max_words)
    # convert DataFrame into dict
    if type(word_freq) == pd.Series:
        counter = Counter(word_freq.fillna(0).to_dict())
    else:
        counter = word_freq
    # filter stop words in frequency counter
    if stopwords is not None:
        counter = {token:freq for (token, freq) in counter.items()
    if token not in stopwords}
        wc.generate_from_frequencies(counter)
    plt.title(title)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")

freq_2015_df = count_words(df[df['year']==2015])
plt.figure()
wordcloud(freq_2015_df['freq'], max_words=100, stopwords=freq.head(0).index)
wordcloud(freq_2015_df['freq'], max_words=100, stopwords=freq.head(50).index)
plt.show()

"""
    TF-IDF (Inverse document frequency explained)
    given some corpus C, the document frequency df(t) is:
    df(t) = |{d in C}|{t in d}|
    terms with a high df(t) appear in many or all documents.

    the IDF is defined as:
    idf(t) = log(|C| / df(t)) + (constant)

    we use log to scale in order to avoid letting really infrequent words matter too much
    some formulas add a constant so as to not completely ignore words that appear a lot.

    tf-idf(t) = tf(t,D) * idf(t) [product of term frequency and the idf of term t]
"""

#defining a formula that computes it.
def compute_idf(df, column='tokens', preprocess=None, min_df=2):
    #update the counter later. if we want to preprocess, do that first, then update the counter.
    def update(doc):
        tokens = doc if preprocess is None else preprocess(doc)
        counter.update(set(tokens))
    # count tokens
    counter = Counter()
    df[column].map(update)
    # create DataFrame and compute idf
    idf_df = pd.DataFrame.from_dict(counter, orient='index', columns=['df'])
    idf_df = idf_df.query('df >= @min_df')
    print(idf_df.head())
    idf_df['idf'] = np.log(len(df)/idf_df['df'])+0.1
    idf_df.index.name = 'token'
    return idf_df

tf_idf = compute_idf(df)

#we are able to do this because both of these dataframes have the same index. 
freq['tfidf'] = freq['freq'] * idf_df['idf']

freq_1970 = count_words(df[df['year'] == 1970])
freq_2015 = count_words(df[df['year'] == 2015])
freq_1970['tfidf'] = freq_1970['freq'] * idf_df['idf']
freq_2015['tfidf'] = freq_2015['freq'] * idf_df['idf']

#wordcloud(freq_df['freq'], title='All years', subplot=(1,3,1))
wordcloud(freq_1970['freq'], title='1970 - TF',stopwords=['twenty-fifth', 'twenty-five'])
plt.show()
wordcloud(freq_2015['freq'], title='2015 - TF', stopwords=['seventieth'])
plt.show()
wordcloud(freq_1970['tfidf'], title='1970 - TF-IDF',stopwords=['twenty-fifth', 'twenty-five', 'twenty', 'fifth'])
plt.show()
wordcloud(freq_2015['tfidf'], title='2015 - TF-IDF',stopwords=['seventieth'])
plt.show()

"""
    WORDS-IN-CONTEXT ANALYSIS
    some words don't make sense. if you look at the last four word clouds, words like PV, SPV, etc. are confusing.
    we can generate context for them.
"""


#create a function that gets KWIC samples
def kwic(doc_series, keyword, window=35, print_samples=5,case_sensitive=False):
    #convert our series into a dataframe for easier operations.
    docs_df = doc_series.to_frame(name='text')
    #coerce everything to lowercase if we get user specification.
    if not case_sensitive:
        keyword = keyword.lower()
        docs_df['text'] = docs_df['text'].apply(lambda x: x.lower())
    #regex logic: as long as they keyword appears in the text with any sort of boundaries, we're interested in a match.
    regex_str = rf'\b{keyword}\b'
    #create a function that returns the span if we find a word, and returns nothing otherwise.
    def get_search_span(regex_str,x):
        search = re.search(regex_str,x)
        if(search is not None):
            return search.span()
        return None
    #apply and get a sample.
    docs_df['regex_match'] = docs_df['text'].apply(lambda x: get_search_span(regex_str,x) )
    #get the number of required samples and shed the rest of our data. we can do this 
    #by seeing if the value in the regex match column is larger than a 0,0 tuple.
    docs_df = docs_df[docs_df['regex_match']>=(0,0)].sample(print_samples)
    #discard the rest of our text after re-wiring the match indices with the window param.
    def add_window(match_tuple,window):
        lower_bound = max(0,match_tuple[0]-window)
        upper_bound = match_tuple[1]+window 
        return((lower_bound,upper_bound))
    #our regex match tuple now reflects an appropriate window around the keyword.
    docs_df['regex_match'] =docs_df['regex_match'].apply(lambda x: add_window(x,window))
    #enclose our keyword in special chars after splicing it.
    def curlify(text):
        return (re.sub(regex_str,f'||{keyword}||',text))
    docs_df['text'] = docs_df.apply(lambda x: x.text[x.regex_match[0]:x.regex_match[1]],axis=1)
    docs_df['text'] = docs_df['text'].map(curlify)
    #set the right column width in pandas to make everything visible.
    pd.set_option("max_colwidth",(2*window)+len(keyword))
    print(docs_df['text'])

    


# test it out
docs = kwic(df[df['year'] == 2015]['text'], 'sdgs', print_samples=5)
