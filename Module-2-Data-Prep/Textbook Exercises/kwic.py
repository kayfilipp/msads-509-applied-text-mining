import pandas as pd
import numpy as np
import nltk
import re 

#create a function that gets KWIC samples
'''
ASSUMPTIONS:
you have a document series with a column called 'text' which contains a string/nvarchar.
'''
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
    docs_df['regex_match'] = docs_df['text'].apply(lambda x: get_search_span(regex_str,x))
    
    #get the number of required samples and shed the rest of our data. we can do this 
    #by seeing if the value in the regex match column is larger than a 0,0 tuple.
    
    #if we get fewer matches than we want to print, we either quit entirely
    docs_df = docs_df[docs_df['regex_match']>=(0,0)]
    if len(docs_df)==0:
        print('no samples')
        return
    else:
        print_samples = min(print_samples,len(docs_df))
    
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