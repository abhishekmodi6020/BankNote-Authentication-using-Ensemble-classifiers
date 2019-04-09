# BankNote-Authentication-using-Ensemble-classifiers

- To predict whether a banknote is authentic or not from a set of real numbered attributes characterizing properties of small image patches taken of the banknote, namely: i)the variance of the Wavelet transformed image, ii) the skewness of the Wavelet transformed image, iii)the curtosis of the Wavelet transformed image, and iv) the entropy of image

- Datasets are derived from the more expansive UCI machine learning banknote data set.

- Implemented a bagging routine for a logistic regression classifier. Applied bagging 10, 50, and 100 times to the training data. 
Evaluated the resulting ensemble classifier on the test data set and compared the error rates for a single classifier and the three ensemble classifiers.

- Implemented AdaBoost on top of logistic regression classifier. Applied boosting 10, 50, and 100 times to the training data. 
Evaluated the resulting ensemble classifier on the test data set and compared the error rates for a single classifier and the three ensemble classifiers.
