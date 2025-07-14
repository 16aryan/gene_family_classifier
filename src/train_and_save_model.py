# test_train_model.py
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from train_model import train_and_save_model

# Dummy encoded k-mer sequences and labels
X_text = [
    "ATG TGC GCA", 
    "TGC GCA CAT", 
    "GCA CAT ATG", 
    "CAT ATG TGC"
]
y = [0, 1, 1, 0]

# Vectorize the text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X_text)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔥 CALL the function
model = train_and_save_model(X_train, y_train, X_test, y_test)