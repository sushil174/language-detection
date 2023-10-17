import pandas as pd
import numpy as np
import sys as sys

class Model:
    def __init__(self) :
        sys.stdin.reconfigure(encoding='utf-8')
        sys.stdout.reconfigure(encoding='utf-8')
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.model_selection import train_test_split
        from sklearn.naive_bayes import MultinomialNB

        data = pd.read_csv(
            "https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv"
        )
        data.isnull().sum()
        data["language"].value_counts()
        x = np.array(data["Text"])
        y = np.array(data["language"])

        self.cv = CountVectorizer()
        X = self.cv.fit_transform(x)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)
        self.model = MultinomialNB()
        self.model.fit(X_train, y_train)
        self.model.score(X_test, y_test)
        
    def detect(self,user):
        data = self.cv.transform([user]).toarray()
        output = self.model.predict(data)
        result = ' '.join(output)
        return result