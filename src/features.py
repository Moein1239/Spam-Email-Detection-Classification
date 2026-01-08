from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf_features(texts, max_features=3000):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1,2),   
        min_df=2             
    )
    X = vectorizer.fit_transform(texts)
    return X, vectorizer
