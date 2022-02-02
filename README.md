## Table Of Content

1. About NB Model
2. Naive Bayes Mind Map
3. Fundamentals of probability
4. NB working
5. NB Classifier on text data
6. Limitations of NB
7. Failure of NB
8. Numerical Stability
9. Bais-Variance trade off for NB
10. Feature Importance in NB
11. Interpretability in NB
12. Imbalanced data in NB
13. Outliers in NB
14. Missing values in NB
15. Can NB do multi-class classification ?
16. Can NB handle large dimentional data ?
17. Best & worst case of NB 
18. Acknowledgements
19. License
20. Connect with me

---

## 1. About NB Model

- Based on fundamentals of probability
- Classification algorithm
- Simplistic & unsophiscated algorithm

## 2.  Naive Bayes Mind Map:
![NB mind mapping](https://github.com/Akshaykumarcp/ML-Model-Naive-Bayes/blob/main/NB.jpg)

# 3. Fundamentals of probability

### 3.1 Conditional probability

![Conditional probability](https://miro.medium.com/max/768/0*AJJVGTHIkEHrvgmH.png)

### 3.2 Independent events &

### 3.3 Dependent events

### 3.4 Derive Bayes theorem from conditional probability & Bayes theorem

[Derive Bayes theorem from conditional probability](https://sites.google.com/site/artificialcortext/others/mathematics/bayes-theorem)

### Bayes theorem

According to the Wikipedia, In probability theory and statistics,* Bayes’s theorem** (alternatively *Bayes’s law or Bayes’s rule) describes the probability of an event, based on prior knowledge of conditions that might be related to the event. Mathematically, it can be written as:

![Conditional probability](https://miro.medium.com/max/1400/1*LB-G6WBuswEfpg20FMighA.png)

Where A and B are events and P(B)≠0
- P(A|B) is a conditional probability: the likelihood of event A occurring given that B is true.
- P(B|A) is also a conditional probability: the likelihood of event B occurring given that A is true.
- P(A) and P(B) are the probabilities of observing A and B respectively; they are known as the marginal probability.

Let’s understand it with the help of an example:

The problem statement:

There are two machines which manufacture bulbs. Machine 1 produces 30 bulbs per hour and machine 2 produce 20 bulbs per hour. Out of all bulbs produced, 1 % turn out to be defective. Out of all the defective bulbs, the share of each machine is 50%. What is the probability that a bulb produced by machine 2 is defective?

We can write the information given above in mathematical terms as:

The probability that a bulb was made by Machine 1, P(M1)=30/50=0.6

The probability that a bulb was made by Machine 2, P(M2)=20/50=0.4

The probability that a bulb is defective, P(Defective)=1%=0.01

The probability that a defective bulb came out of Machine 1, P(M1 | Defective)=50%=0.5

The probability that a defective bulb came out of Machine 2, P(M2 | Defective)=50%=0.5

Now, we need to calculate the probability of a bulb produced by machine 2 is defective i.e., P(Defective | M2). Using the Bayes Theorem above, it can be written as:

P(Defective|M2) = P(M2|Defective) ∗ P(Defective) / P(M2)

Substituting the values, we get:

P(Defective|M2) = 0.5 ∗ 0.01 / 0.4 = 0.0125

## 4. NB Working

- https://datacadamia.com/data_mining/naive_bayes

## 5. NB Classifier on text data

## 6. Limitation of NB

- When a new word is provided in test query which is not present in training data, likelyhood of new word cannot be computed.
- what are the possibilities to handle ?
- how about dropping the new word ? Dropping is equivalent to saying likelyhood is equal to 1
- If likelyhood is assigned to 0, the entire bayes theorem multiplication results into 0
- Therefore, value 1 or 0 doesn't make sense!!
- We need to have a better scheme to handle i,e Laplace Smoothing / Additive Smoothing

### Laplace Smoothing / Additive Smoothing

- https://en.wikipedia.org/wiki/Additive_smoothing
- alpha is the hyperparameter. When alpha is large, approximatly 1/2 (half) value will be assigned to the likelyhood.
- 1/2 (half) value is better because its good to say 1/2 to the untrained/new word during test time.
- Called smoothing because as the value of alpha increases, the likelyhood is decreased gradually. The decreasing value is known as smoothing
- Note: Laplace smoothing will be applied not only during test time, also during training time 

## 8. Numerical Stability

- when training dataset dimensionality is large and multiplication happens between 0 to 1 in probability, we end up getting very small number. Ex: 0.0004
- To avoid numerical stability issue in NB, instead of operating on small values operate on log values

## 9. Bais-Variance trade off for NB

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

### How to choose right alpha value :question:

- Hyperparameter tuning using cross validation

## 10. Feature Importance in NB

- In NB, likelyhood of all words are computed
- Sort the likelyhood of words in descending order and pick top "n" words for feature importance

## 11. Interpretability in NB

- For the predicted class, we shall provide "n" (Ex: word1, word2, . . . . wordn ) words as evidence
- Ex: for classification of positive and negative review
- Phenomenal, great and terrific are the high occurence for positive review

## 12. Imbalanced data in NB

- Class prior which are majority/dominating class have an advantage when comparing two probablties
- Hence majority class will be predicted at prediction time
- NB is affected from imbalanced data

### Solution :bulb:

- Upsampling
- Downsampling

## 13. Outliers in NB

- Find the words that fewer occurs in training data for outlier
- When outlier is present in testing time, laplace smoothing will take care of it

### Solution :bulb:

- Ignore rare words
- Use laplace smoothing

## 14. Missing values in NB

- text data: there is no case of missing data in case of text data
- categorical feature: consider "NAN" as a category
- Numerical feature: impute mean, median, etc

## 15. Can NB do multi-class classification :question:

- NB supports multi-class classification.
- compares against all the probabilities and returns the maximum probability

## 16. Can NB handle large dimentional data :question:

- NB does text classification i,e high dimentional capability
- So NB is able to handle large dim data

**Note:** Make sure to use log probabilities in order to avoid nmerical overflow or stability underflow issue

## 17. Best & worst case of NB

### 17.1 Conditional independence of features

- When true; NB performs well
- When false; NB performance degrades

### 17.2 Some features are dependent

- NB works fairly well

### 17.3 Text classification

- email/review classification: high dimentional data
- NB works well &
- NB is the baseline/benchmark model

### 17.4 Often used when categorical features

### 17.5 Real value features

- seldom or not used much

### 17.6 NB is interpretable & provides feature importance

### 17.7 Run time/ train time/ space are low

- because store only prior and likelihood probabilities
- NB is all about counting

### 17.8 Easily overfit if laplace smoothing is not done

## 18. Acknowledgements :handshake:

 - [Google Images](https://www.google.co.in/imghp?hl=en-GB&tab=ri&authuser=0&ogbl)
  
## 19. License :page_facing_up:

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)

## 20. Connect with me :smiley:
[<img align="left" alt="" width="22px" src="https://simpleicons.org/icons/linkedin.svg" />](https://www.linkedin.com/in/akshay-kumar-c-p/)
[<img align="left" alt="" width="22px" src="https://simpleicons.org/icons/youtube.svg" />](https://www.youtube.com/channel/UC3l8RTE3zBRzUrHbSXpx-qA)
[<img align="left" alt="" width="22px" src="https://simpleicons.org/icons/github.svg" />](https://github.com/Akshaykumarcp)
[<img align="left" alt="" width="22px" src="https://simpleicons.org/icons/medium.svg" />](https://medium.com/@akshai.148)
