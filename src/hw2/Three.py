from operator import attrgetter

import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

from hw2.domain.ProviderSortedJayCodes import ProviderSortedJayCodes
from hw2.util.ClassifiersExecutor import ClassifiersExecutor


class Three:

    def __init__(self, provider_sorted_jay_codes: ProviderSortedJayCodes, use_subset):
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

    def a(self):
        # What percentage of J-code claim lines were unpaid?
        return len(self.unpaid_claims_L) / len(self.all_claims_w_L) * 100

    def init_model_data(self):
        # Create a model to predict when a J-code is unpaid. Explain why you choose the modeling approach.

        # 1- We need to shuffle the rows
        self.all_claims_w_L.dtype.names
        # Apply the random shuffle
        np.random.shuffle(self.all_claims_w_L)
        self.all_claims_w_L.dtype.names

        label = 'IsUnpaid'
        numeric_features = ['ClaimNumber', 'ClaimLineNumber', 'ClaimChargeAmount']

        categoricals = ['RevenueCode', 'PriceIndex', 'InOutOfNetwork',
                        'NetworkID', 'ClaimType']

        categorical_features = np.array(
            self.all_claims_w_L[categoricals].tolist())  # convert features to list, then to np.array
        numerical_features = np.array(
            self.all_claims_w_L[numeric_features].tolist())  # convert features to list, then to np.array

        label_encoder = preprocessing.LabelEncoder()
        for i in range(len(categoricals)):
            categorical_features[:, i] = label_encoder.fit_transform(categorical_features[:, i])

        OneHotEncoder = preprocessing.OneHotEncoder(sparse=False)
        categorical_features = OneHotEncoder.fit_transform(categorical_features)

        M = np.concatenate((numerical_features, categorical_features), axis=1)  # Concatenate the columns
        L = self.all_claims_w_L[label].astype(int)
        M.shape
        L.shape
        np.corrcoef(M)
        self.data = (M, L, 5)

    def b_and_c(self):
        # 3B. Create a model to predict when a J-code is unpaid. Explain why you choose the modeling approach.
        # 3C. How accurate is your model at predicting unpaid claims? this method generates a png file in the result folder
        executor = ClassifiersExecutor()
        classification_output = executor.execute(self.data)
        print('Algorithms with different hyper parameters and the accuracy result:')
        for output_item in classification_output:
            print(output_item.description())
        model_with_highest_accuracy = max(classification_output, key=attrgetter('accuracy'))
        print('3B and C. The classification Result stemmed from reusing homework 1 classifiers illustrates '
              'that KNN predicts unpaid claims better than Random Forest and other classifiers: ')
        print(model_with_highest_accuracy.description())
        return classification_output

    def d(self):
        # What data attributes are predominately influencing the rate of non-payment?
        column_names = ['ClaimNumber', 'ClaimLineNumber', 'MemberID',
                        'ClaimChargeAmount', 'SubscriberPaymentAmount', 'ProviderPaymentAmount',
                        'GroupIndex', 'SubscriberIndex', 'SubgroupIndex', 'V1', 'ProviderID', 'LineOfBusinessID',
                        'RevenueCode', 'ServiceCode', 'PlaceOfServiceCode', 'ProcedureCode', 'DiagnosisCode',
                        'DenialReasonCode', 'PriceIndex', 'InOutOfNetwork', 'ReferenceIndex', 'PricingIndex',
                        'CapitationIndex', 'ClaimSubscriberType', 'ClaimPrePrinceIndex', 'ClaimCurrentStatus',
                        'NetworkID', 'AgreementID', 'ClaimType'
                        ]

        M, L, n_folds = self.data
        classifier = RandomForestClassifier(n_estimators='warn', criterion='gini', min_samples_split=6,
                                            max_features='auto', n_jobs=-1)
        classifier.fit(M, L)
        feature_importance_map = zip(column_names, classifier.feature_importances_)
        print('3.D I am using Random Forest Classifer to answer 3d. It is the second best algorithm among all my '
              'classifiers. The model suggests that PriceIndex, ClaimType, RevenueCode, InOutOfNetwork, '
              'NetworkId, ClaimChargeAmount have the highest influence. Considering that our model'
              ' is learning based on Jcodes, we should avoid over fitting by not adding too many variables '
              'into the model. We should also be very cautious about sequential identifiers and redundant features '
              '(e.g. DenialReasonCode and perhaps ProcedureCode) as model with limited data might suggest influence'
              ' while in reality the correlation might not make any sense.')

        for item in feature_importance_map:
            print(item)
        return feature_importance_map
