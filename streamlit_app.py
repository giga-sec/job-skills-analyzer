import streamlit as st
from pandas import DataFrame
import pandas as pd
from plotly.express import line, bar
import boto3
import io

import nltk
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download('punkt_tab')

def tokenize_lemmatize(text):
    from re import sub
    from string import punctuation
    from nltk.corpus import stopwords
    