"""
example for Online LDA with 20 news group
"""

# Authors: Chyi-Kwei Yau


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from lda import LDA


def main():
    n_samples = 2000
    n_features = 1000
    n_topics = 20
    n_top_words = 15

    dataset = fetch_20newsgroups(
        shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))

    vectorizer = CountVectorizer(token_pattern=r"(?u)\b[^\d\W]\w+\b",
                                 max_df=0.9, max_features=n_features, min_df=2, stop_words='english')

    doc_word_count = vectorizer.fit_transform(dataset.data[:n_samples])
    lda = LDA(n_topics=n_topics, kappa=0.7,
              tau0=1024., n_jobs=4, random_state=0)

    feature_names = vectorizer.get_feature_names()
    lda.fit(doc_word_count)
    for topic_idx, topic in enumerate(lda.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


if __name__ == '__main__':
    main()
