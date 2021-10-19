# ML-Model-Naive-Bayes
Naive Bayes Machine Learning Model

# Content

1. About NB Model
2. NB Mind Map image
3. Fundamentals of probability
4. NB working
5. Distance measures
6. Limitations of NB
7. Failure of NB
8. NB techniques to reduce space & time complexity
9. Decision surface for NB
10. KNN when Overfit & underfit
11. Can NB do Multiclass classification ?
12. Can NB do Probabilitics predictions ?
13. NB when dataset is imbalanced
14. NB when dataset has outliers
16. Is NB affected by scale of the features ?
17. Are NB interpetable ?
18. Acknowledgements
19. License

# About NB Model

- Based on fundamentals of probability
- Classification algorithm
- Simplistic & unsophiscated algorithm

# Naive Bayes Mind Map:
![NB mind mapping](https://github.com/Akshaykumarcp/ML-Model-Naive-Bayes/blob/main/NB.jpg)

# Fundamentals of probability

## 1. Conditional probability

![Conditional probability](https://miro.medium.com/max/768/0*AJJVGTHIkEHrvgmH.png)

## 2. Independent events &

## 3. Dependent events

# Derive Bayes theorem from conditional probability

![Bayes theorem](https://sites.google.com/site/artificialcortext/others/mathematics/bayes-theorem?tmpl=%2Fsystem%2Fapp%2Ftemplates%2Fprint%2F&showPrintDialog=1)

# NB Working

- https://datacadamia.com/data_mining/naive_bayes

# NB Classifier on text data

# Limitation of NB

- When a new word is provided in test query which is not present in training data, likelyhood of new word cannot be computed.
- what are the possibilities to handle ?
- how about dropping the new word ? Dropping is equivalent to saying likelyhood is equal to 1
- If likelyhood is assigned to 0, the entire bayes theorem multiplication results into 0
- Therefore, value 1 or 0 doesn't make sense!!
- We need to have a better scheme to handle i,e Laplace Smoothing / Additive Smoothing

## Laplace Smoothing / Additive Smoothing

- https://en.wikipedia.org/wiki/Additive_smoothing
- alpha is the hyperparameter. When alpha is large, approximatly 1/2 (half) value will be assigned to the likelyhood.
- 1/2 (half) value is better because its good to say 1/2 to the untrained/new word during test time.
- Called smoothing because as the value of alpha increases, the likelyhood is decreased gradually. The decreasing value is known as smoothing
- Note: Laplace smoothing will be applied not only during test time, also during training time 

# Numerical Stability

- when training dataset dimensionality is large and multiplication happens between 0 to 1 in probability, we end up getting very small number. Ex: 0.0004
- To avoid numerical stability issue in NB, instead of operating on small values operate on log values

# Bais-Variance trade off for Naive Bayes

- High Bias --> Underfitting
- High Variance --> Overfitting

alpha in laplace smoothing

### Overfitting

- When alpha is 0;
- for rare words in training data, probability is computed
- when the rare words are removed in training data, 0 probability is returned.
- So when there is small change in data, there is large impact on model predictions. 
- Is the high variance and results in overfitting

### Underfitting

- When alpha is very large; i,e alpha = 10,000
- Approximately 1/2 (half) is the probability
- All the prediction is predicted as 1/2. Model cannot distinguish between 0 or 1
- Is the high bias and results in underfitting

Therefore Bias & Variance trade off depends on alpha value

### How to choose right alpha value ?

- Hyperparameter tuning using cross validation

# Feature Importance 

- In NB, likelyhood of all words are computed
- Sort the likelyhood of words in descending order and pick top "n" words for feature importance

# Interpretability

- For the predicted class, we shall provide "n" (Ex: word1, word2, . . . . wordn ) words as evidence
- Ex: for classification of positive and negative review
- Phenomenal, great and terrific are the high occurence for positive review

# Imbalanced data

- Class prior which are majority/dominating class have an advantage when comparing two probablties
- Hence majority class will be predicted at prediction time
- NB is affected from imbalanced data

### Solution ?

- Upsampling
- Downsampling

# Acknowledgements

 - [Google Images](https://www.google.co.in/imghp?hl=en-GB&tab=ri&authuser=0&ogbl)
  
# License

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
