import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
import re

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


port = PorterStemmer()


def clean_text(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    lmtzr = WordNetLemmatizer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = [lmtzr.lemmatize(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    return review


def main():
    st.title("Positive Review Vs Negative Rating Detection")
    st.markdown(
        "Identify such ratings where review text is good, but rating is negative."
        " So that the support team can point this to users.")
    st.text("Note: Data should be in csv format.")

    uploaded_file = st.file_uploader("Choose a File", type=["csv"])

    df = pd.read_csv(uploaded_file)
    st.write(df)

    if st.button("Positive Negative Semantic"):
        df["cleaned_text"] = df["Text"].apply(lambda x: clean_text(str(x)))

        sid_obj = SentimentIntensityAnalyzer()

        df["sentiment_dict"] = df["cleaned_text"].apply(lambda text: sid_obj.polarity_scores(text))
        df['compound_score'] = df['sentiment_dict'].apply(lambda x: x['compound'])
        df['Review'] = df['compound_score'].apply(
            lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral'))
        positive_review = df[df['Review'] == 'positive']
        positive_review['Alert'] = positive_review['Star'].apply(lambda x: 1 if x <= 2 else 0)
        ax = sns.countplot(x='Alert', data=positive_review)
        for p in ax.patches:
            ax.annotate('{:}'.format(p.get_height()), (p.get_x() + 0.25, p.get_height() + 0.01))
        alert = positive_review.Alert.value_counts()
        if alert[1] == 0:
            st.text("No user has entered wrong rating.")
        else:
            st.text(f'{alert[1]} users have given positive review.')

    data = df_positive

    st.download_button(
        label="Download data as CSV",
        data=data.to_csv().encode("utf-8"),
        file_name='data.csv',
        mime='text/csv',
    )


main()
