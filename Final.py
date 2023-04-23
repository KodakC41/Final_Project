import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
#from textblob import TextBlob
import pandas as pd
import numpy as np
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize


sentiment_analysis_columns = ['negativity', 'neutrality', 'positivity', 'compound']

file_names = [...]

sid = SentimentIntensityAnalyzer()

for i in range(len(file_names)):
    expanded_text_dataset = []
    file_name = file_names[i]
    print('Sentiment analysis for: %s' %file_name)
    file_df = pd.read_excel('%s.xlsx' % file_name)
    column_names = list(file_df.columns.values)
    all_columns = column_names + sentiment_analysis_columns
    text_segment_list = list(file_df['text_segment'])
    for j in range(len(text_segment_list)):
        row_info = list(file_df.iloc[j])
        text = text_segment_list[j]
        ##Sentiment analysis
        try:
            ss = sid.polarity_scores(text)
            negativity = ss['neg']
            neutrality = ss['neu']
            positivity = ss['pos']
            compound = ss['compound']
            temp_data = row_info + [negativity, neutrality, positivity, compound]
            expanded_text_dataset.append(temp_data)
        except:
            print('Sentiment analysis not done: ' + str(i) + ' / ' + str(len(file_df)))
            temp_data = row_info + ['NA', 'NA', 'NA', 'NA']
            expanded_text_dataset.append(temp_data)
    ##Storing the results in a dataset
    sentiment_dataset_df = pd.DataFrame(expanded_text_dataset, columns = all_columns)
    sentiment_dataset_df.to_excel('%s_sentiment_analysis.xlsx' %file_name)