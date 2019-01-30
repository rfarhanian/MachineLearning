# adapt this code below to run your analysis

# Due before live class 2
# 1. Write a function to take a list or dictionary of clfs and hypers ie use logistic regression, each with 3 different sets of hyper parrameters for each

# Due before live class 3
# 2. expand to include larger number of classifiers and hyper parameter settings
# 3. find some simple data
# 4. generate matplotlib plots that will assist in identifying the optimal clf and parameters settings

# Due before live class 4
# 5. Please set up your code to be run and save the results to the directory that its executed from
# 6. Investigate grid search function

``` python
def run(a_clf, data, clf_hyper={}):
    M, L, n_folds = data  # unpack data container
    kf = KFold(n_splits=n_folds)  # Establish the cross validation
    ret = {}  

    for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
        clf = a_clf(**clf_hyper)  # unpack parameters into clf is they exist

        clf.fit(M[train_index], L[train_index])

        pred = clf.predict(M[test_index])

        ret[ids] = {'clf': clf,
                    'train_index': train_index,
                    'test_index': test_index,
                    'accuracy': accuracy_score(L[test_index], pred),
                    'classificationReport': classification_report(L[test_index], pred)
                    }
    return ret

```