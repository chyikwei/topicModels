"""
example for Online LDA with 20 news group
"""

# Authors: Chyi-Kwei Yau


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from lda import onlineLDA


def main():
    n_samples = 4000
    n_features = 1000
    n_top_words = 15

    dataset = fetch_20newsgroups(
        shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))

    vectorizer = CountVectorizer(max_df=0.8, max_features=n_features, min_df=3, stop_words='english')

    doc_word_count = vectorizer.fit_transform(dataset.data[:n_samples])
    lda = onlineLDA(kappa=0.7, tau=512., n_jobs=-1, random_state=0, verbose=1)

    feature_names = vectorizer.get_feature_names()
    lda.fit(doc_word_count, max_iters=20)
    for topic_idx, topic in enumerate(lda.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

def lda_simple_example():
    # TODO: complete test
    from sklearn.feature_extraction.text import CountVectorizer

    test_words = ['aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii', 'jj']
    test_vocab = {}
    for idx, word in enumerate(test_words):
        test_vocab[word] = idx

    # group 1: aa, bb, cc, dd
    # group 2: ee ff gg
    # group 3: hh ii jj
    n_topics = 3
    test_docs = ['aa bb cc dd aa aa',
                 'ee ee ff ff gg gg',
                 'hh ii hh ii jj jj jj jj',
                 'aa bb cc dd aa aa dd aa bb cc',
                 'ee ee ff ff gg gg',
                 'hh ii hh ii jj jj jj jj',
                 'aa bb cc dd aa aa',
                 'ee ee ff ff gg gg',
                 'hh ii hh ii jj jj jj jj',
                 'aa bb cc dd aa aa dd aa bb cc',
                 'ee ee ff ff gg gg',
                 'hh ii hh ii jj jj jj jj']

    n_top_words = 4
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b[^\d\W]\w+\b",
                                 max_df=0.9, min_df=1, vocabulary=test_vocab)

    doc_word_count = vectorizer.fit_transform(test_docs)
    lda = onlineLDA(n_topics=n_topics, kappa=0.7,
              tau0=1024., random_state=0, n_jobs=-1)
    lda.fit(doc_word_count)
    feature_names = vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(lda.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

if __name__ == '__main__':
    main()
