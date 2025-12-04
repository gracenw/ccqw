import numpy as np
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from data import read_train_data
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

np.random.seed(777)


def main():
    texts, labels = read_train_data('/home/gracen/repos/peace/data')

    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()

    train_texts_counts = count_vect.fit_transform(train_texts)
    train_texts_tfidf = tfidf_transformer.fit_transform(train_texts_counts)

    val_texts_counts = count_vect.transform(val_texts)
    val_texts_tfidf = tfidf_transformer.transform(val_texts_counts)

    # naive_bayes = MultinomialNB().fit(train_texts_tfidf, train_labels)
    # nb_predictions = naive_bayes.predict(val_texts_tfidf)
    # nb_accuracy = np.mean(nb_predictions == val_labels)
    # print('naive bayes accuracy:', nb_accuracy)

    C_RANGE = np.logspace(-3, 3, 20)
    G_RANGE = np.logspace(-9, 1, 20)
    param_grid = { 'C': C_RANGE,
                   'gamma': ['scale', 'auto'],
                   'gamma': G_RANGE,
                   'degree': [2, 3, 4],
                   'kernel': ['rbf', 'poly'], 
                   'coef0': [0, 1],
                   'class_weight': ['balanced'] }

    svm = GridSearchCV(SVC(), param_grid, n_jobs = -1, verbose = 0)
    svm.fit(train_texts_tfidf, train_labels)

    Copt = svm.best_params_['C'] ## svm cost parameter
    Kopt = svm.best_params_['kernel'] ## kernel function
    Gopt = svm.best_params_['gamma'] ## gamma of RBF kernel
    Dopt = svm.best_params_['degree'] ## degree of polynomial kernel
    Zopt = svm.best_params_['coef0'] ## independent term in poly kernel

    print('\nOptimal SVM parameter values:')
    print('C:', Copt)
    print('kernel:', Kopt)
    print('gamma:', Gopt)
    print('degree:', Dopt)
    print('coef0:', Zopt, '\n')

    print('Calculating metrics...') ## generate report
    svm_predictions = svm.predict(val_texts_tfidf)
    print(classification_report(val_labels, svm_predictions))
    scores = cross_val_score(svm, val_texts_tfidf, val_labels, cv = 6)
    print('\nAverage cross-validate score: ', scores.mean())

    # svm = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=15, tol=None).fit(train_texts_tfidf, train_labels)
    # svm_predictions = svm.predict(val_texts_tfidf)
    # svm_accuracy = np.mean(svm_predictions == val_labels)
    # print('svm accuracy:', svm_accuracy)


if __name__ == '__main__':
    main()