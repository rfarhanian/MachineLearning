from itertools import product

import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

from hw2.domain.ProviderSortedJayCodes import ProviderSortedJayCodes


class Three:

    def __init__(self, provider_sorted_jay_codes: ProviderSortedJayCodes):
        all_claim_codes = provider_sorted_jay_codes.get_jay_codes()
        unpaid_claims_mask = (all_claim_codes['ProviderPaymentAmount'] == 0)
        paid_claims_mask = (all_claim_codes['ProviderPaymentAmount'] > 0)

        unpaid_claims = all_claim_codes[unpaid_claims_mask]
        paid_claims = all_claim_codes[paid_claims_mask]
        unpaid_claims_dtype = np.dtype(unpaid_claims.dtype.descr + [('IsUnpaid', '<i4')])
        paid_claims_dtype = np.dtype(paid_claims.dtype.descr + [('IsUnpaid', '<i4')])
        self.unpaid_claims_L = np.zeros(unpaid_claims.shape, dtype=unpaid_claims_dtype)
        self.paid_claims_L = np.zeros(paid_claims.shape, dtype=paid_claims_dtype)

        self.unpaid_claims_L.shape
        self.paid_claims_L.shape

        # The following code can be optimized by having a list of attributes in the ClaimParser class
        self.unpaid_claims_L['V1'] = unpaid_claims['V1']
        self.unpaid_claims_L['ClaimNumber'] = unpaid_claims['ClaimNumber']
        self.unpaid_claims_L['ClaimLineNumber'] = unpaid_claims['ClaimLineNumber']
        self.unpaid_claims_L['MemberID'] = unpaid_claims['MemberID']
        self.unpaid_claims_L['ProviderID'] = unpaid_claims['ProviderID']
        self.unpaid_claims_L['LineOfBusinessID'] = unpaid_claims['LineOfBusinessID']
        self.unpaid_claims_L['RevenueCode'] = unpaid_claims['RevenueCode']
        self.unpaid_claims_L['ServiceCode'] = unpaid_claims['ServiceCode']
        self.unpaid_claims_L['PlaceOfServiceCode'] = unpaid_claims['PlaceOfServiceCode']
        self.unpaid_claims_L['ProcedureCode'] = unpaid_claims['ProcedureCode']
        self.unpaid_claims_L['DiagnosisCode'] = unpaid_claims['DiagnosisCode']
        self.unpaid_claims_L['ClaimChargeAmount'] = unpaid_claims['ClaimChargeAmount']
        self.unpaid_claims_L['DenialReasonCode'] = unpaid_claims['DenialReasonCode']
        self.unpaid_claims_L['PriceIndex'] = unpaid_claims['PriceIndex']
        self.unpaid_claims_L['InOutOfNetwork'] = unpaid_claims['InOutOfNetwork']
        self.unpaid_claims_L['ReferenceIndex'] = unpaid_claims['ReferenceIndex']
        self.unpaid_claims_L['PricingIndex'] = unpaid_claims['PricingIndex']
        self.unpaid_claims_L['CapitationIndex'] = unpaid_claims['CapitationIndex']
        self.unpaid_claims_L['SubscriberPaymentAmount'] = unpaid_claims['SubscriberPaymentAmount']
        self.unpaid_claims_L['ProviderPaymentAmount'] = unpaid_claims['ProviderPaymentAmount']
        self.unpaid_claims_L['GroupIndex'] = unpaid_claims['GroupIndex']
        self.unpaid_claims_L['SubscriberIndex'] = unpaid_claims['SubscriberIndex']
        self.unpaid_claims_L['SubgroupIndex'] = unpaid_claims['SubgroupIndex']
        self.unpaid_claims_L['ClaimType'] = unpaid_claims['ClaimType']
        self.unpaid_claims_L['ClaimSubscriberType'] = unpaid_claims['ClaimSubscriberType']
        self.unpaid_claims_L['ClaimPrePrinceIndex'] = unpaid_claims['ClaimPrePrinceIndex']
        self.unpaid_claims_L['ClaimCurrentStatus'] = unpaid_claims['ClaimCurrentStatus']
        self.unpaid_claims_L['NetworkID'] = unpaid_claims['NetworkID']
        self.unpaid_claims_L['AgreementID'] = unpaid_claims['AgreementID']

        self.unpaid_claims_L['IsUnpaid'] = 1

        # The following code can be optimized by having a list of attributes in the ClaimParser class
        self.paid_claims_L['V1'] = paid_claims['V1']
        self.paid_claims_L['ClaimNumber'] = paid_claims['ClaimNumber']
        self.paid_claims_L['ClaimLineNumber'] = paid_claims['ClaimLineNumber']
        self.paid_claims_L['MemberID'] = paid_claims['MemberID']
        self.paid_claims_L['ProviderID'] = paid_claims['ProviderID']
        self.paid_claims_L['LineOfBusinessID'] = paid_claims['LineOfBusinessID']
        self.paid_claims_L['RevenueCode'] = paid_claims['RevenueCode']
        self.paid_claims_L['ServiceCode'] = paid_claims['ServiceCode']
        self.paid_claims_L['PlaceOfServiceCode'] = paid_claims['PlaceOfServiceCode']
        self.paid_claims_L['ProcedureCode'] = paid_claims['ProcedureCode']
        self.paid_claims_L['DiagnosisCode'] = paid_claims['DiagnosisCode']
        self.paid_claims_L['ClaimChargeAmount'] = paid_claims['ClaimChargeAmount']
        self.paid_claims_L['DenialReasonCode'] = paid_claims['DenialReasonCode']
        self.paid_claims_L['PriceIndex'] = paid_claims['PriceIndex']
        self.paid_claims_L['InOutOfNetwork'] = paid_claims['InOutOfNetwork']
        self.paid_claims_L['ReferenceIndex'] = paid_claims['ReferenceIndex']
        self.paid_claims_L['PricingIndex'] = paid_claims['PricingIndex']
        self.paid_claims_L['CapitationIndex'] = paid_claims['CapitationIndex']
        self.paid_claims_L['SubscriberPaymentAmount'] = paid_claims['SubscriberPaymentAmount']
        self.paid_claims_L['ProviderPaymentAmount'] = paid_claims['ProviderPaymentAmount']
        self.paid_claims_L['GroupIndex'] = paid_claims['GroupIndex']
        self.paid_claims_L['SubscriberIndex'] = paid_claims['SubscriberIndex']
        self.paid_claims_L['SubgroupIndex'] = paid_claims['SubgroupIndex']
        self.paid_claims_L['ClaimType'] = paid_claims['ClaimType']
        self.paid_claims_L['ClaimSubscriberType'] = paid_claims['ClaimSubscriberType']
        self.paid_claims_L['ClaimPrePrinceIndex'] = paid_claims['ClaimPrePrinceIndex']
        self.paid_claims_L['ClaimCurrentStatus'] = paid_claims['ClaimCurrentStatus']
        self.paid_claims_L['NetworkID'] = paid_claims['NetworkID']
        self.paid_claims_L['AgreementID'] = paid_claims['AgreementID']

        # And assign the target label
        self.paid_claims_L['IsUnpaid'] = 0
        # here we just combine rows
        self.all_claims_w_L = np.concatenate((self.unpaid_claims_L, self.paid_claims_L), axis=0)
        self.init_model_data()
        self.clfsAccuracyDict = {}

    def a(self):
        # What percentage of J-code claim lines were unpaid?
        return len(self.unpaid_claims_L) / len(self.all_claims_w_L) * 100

    def init_model_data(self):
        # Create a model to predict when a J-code is unpaid. Explain why you choose the modeling approach.

        # We need to shuffle the rows before using classifers in sklearn
        self.all_claims_w_L.dtype.names
        # Apply the random shuffle
        np.random.shuffle(self.all_claims_w_L)

        self.all_claims_w_L.dtype.names

        label = 'IsUnpaid'
        # I just removed columns that are unique or have significant missings or redundant in our model
        # Also by running a Random Forest Feature Importance selection, I found columns have the lowest importance
        # and filtered them out.
        numeric_features = ['ClaimNumber', 'ClaimLineNumber', 'ClaimChargeAmount', 'SubscriberIndex', 'SubgroupIndex', ]

        categorical_features = [
            'ProviderID', 'LineOfBusinessID',  # 'RevenueCode',
            'ServiceCode',  # 'PlaceOfServiceCode', 'ProcedureCode', 'DiagnosisCode',  # 'DenialReasonCode',
            # 'PriceIndex', 'InOutOfNetwork', 'ReferenceIndex',
            'PricingIndex', 'CapitationIndex', 'ClaimSubscriberType',
            # 'ClaimPrePrinceIndex', 'ClaimCurrentStatus',
            'NetworkID',
            'AgreementID', 'ClaimType']

        # convert features to list, then to np.array
        # separate categorical and numeric features
        Mcat = np.array(self.all_claims_w_L[categorical_features].tolist())
        Mnum = np.array(self.all_claims_w_L[numeric_features].tolist())
        L = np.array(self.all_claims_w_L[label].tolist())

        # Run the Label encoder
        le = preprocessing.LabelEncoder()
        for i in range(len(categorical_features)):
            Mcat[:, i] = le.fit_transform(Mcat[:, i])

        # Run the OneHotEncoder
        OneHotEncoder = preprocessing.OneHotEncoder(sparse=False)
        Mcat = OneHotEncoder.fit_transform(Mcat)

        # For start and to avoid getting memory error, I subset the data

        # Mcat_subset = Mcat[0:500]
        # Mcat_subset.shape
        #
        # Mnum_subset = Mnum[0:500]
        # Mnum_subset.shape

        M = np.concatenate((Mnum, Mcat), axis=1)  # Concatenate the columns
        L = self.all_claims_w_L[label].astype(int)

        M.shape
        L.shape

        np.corrcoef(M)

        self.data = (M, L, 5)

        # *************************************************************************
        self.all_models = [RandomForestClassifier, KNeighborsClassifier, LogisticRegression, GradientBoostingClassifier]

        self.hyper_param_mapping = {
            'RandomForestClassifier': {"min_samples_split": [2], "n_jobs": [-1], "n_estimators": [10, 100, 300]},
            'LogisticRegression': {"tol": [0.1], "C": [0.001], "penalty": ["l1"]},
            'KNeighborsClassifier': {"n_neighbors": [4, 6], "n_jobs": [-1], "leaf_size": [20], "algorithm": ["auto"]},
            'GradientBoostingClassifier': {"min_samples_split": [4], "min_weight_fraction_leaf": [0.3]}
        }

        return None

    def run(self, clfs, data, clf_hyper={}):
        M, L, n_folds = data
        kf = KFold(n_splits=n_folds)
        result = {}

        for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
            clf = clfs(**clf_hyper)
            clf.fit(M[train_index], L[train_index])
            pred = clf.predict(M[test_index])
            result[ids] = {'clf': clf, 'accuracy': accuracy_score(L[test_index], pred)}
        return result

    def init_accuracy_mapping(self, Output):
        # Create a dictionary containing the result of accuracy for each set
        for key in Output:
            k1 = Output[key]['clf']
            v1 = Output[key]['accuracy']
            k1Test = str(k1)

            k1Test = k1Test.replace('                    ', '  ')
            k1Test = k1Test.replace('               ', '  ')
            if k1Test in self.clfsAccuracyDict:
                self.clfsAccuracyDict[k1Test].append(v1)
            else:
                self.clfsAccuracyDict[k1Test] = [v1]

    def run_all(self):
        # this function will get the results of each hyper parameter combinations and use for run function
        for clf in self.all_models:
            # to check if values in ModelsList are in ParamDict
            clf_str = str(clf)
            for key, value in self.hyper_param_mapping.items():  # go through the inner dictionary of hyper parameters
                if key in clf_str:
                    # to do all the matching key and values
                    k2, v2 = zip(*value.items())
                    for values in product(*v2):  # for the values in the inner dictionary, get their unique combinations
                        hyperSet = dict(zip(k2, values))  # create a dictionary from their values
                        Output = self.run(clf, self.data, hyperSet)
                        self.init_accuracy_mapping(Output)
                        # print(Output)

    def b(self):
        self.run_all()
        return self.clfsAccuracyDict
