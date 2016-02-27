# Predict Positive or Negative dialogue act

## Dataset Peculiarities
* The first peculiarity is that the dataset is quite inbalanced. There are ~80.000 examples and only ~7% have NEGATIVE_TIME
   label.
* Second, a sentence can have more than one Time entity with different
  dialogue acts. For example: *I'm not available on Tuesday, but
Wednesday works for me.*

## Features
* Unigrams: The first feature included are unigrams as bag-of-words. It only
  includes the presence of a word.
* POS-TAG: The POS TAG of the 2 previous and 2 following words to the
  Time Entity are included as features.
* Negation: Negation should be an important feature, since it changes
  the meaning of a word. Therefore, whenever there is a negation in a
sentence, the *not_* is prepended to the word that are to the right of
the negation. For example, *I can't meet on DATE* it turns to *I can't
not_meet not_on not_Date*

There are more advanced features that can be added like
* Dependency trees
* Lexicons

## Classifier
There should be a lot of features that doesn't add relevant information
to the model, therefore I think a classifier with a L1 penalty can have
a good result as I expect that the model is sparse.

I used Naive Bayes, linear SVM with L1 penalty and linear SVM with L2
penalty and chosed the best among those using cross validation.

## Process

The first thing I did was to preprocess the data and get the features
for the entire dataset. Tokenization, POS-TAGGING and Lemmatization
was applied to the dataset and the result saved into mongo. This process
took a lot of processing power, I had to use a cluster of 20 AWS EC2
instances in order to get the results in less than 3 hours.

Next I separated the dataset into development and
testing. 80% of the dataset was used for development and the rest for
testing.
The development dataset was divided into training and validation, and
again 80% of the development dataset was used for training and the rest
for validation.

As a baseline I trained Naive Bayes, SVM L1, SVM L@ classifiers with the training set
and evaluated it on the validation set.

|Naive Bayes | Precision | Recall | F-measure|
|------------|-----------|--------|----------|
|Positive | 0.983928065704 | 0.871631368778 | 0.92438167245 |
|Negative | 0.329474718794 | 0.815846403461 | 0.469389342668 |


|SVM L1 | Precision | Recall | F-measure|
|------------|-----------|--------|----------|
|Positive | 0.953732744671 |  0.998118375112 | 0.975420889179 |
|Negative | 0.938858695652 | 0.373715521904 | 0.534622823985 |

|SVM L2 | Precision | Recall | F-measure|
|------------|-----------|--------|----------|
|Positive | 0.955316209886 | 0.997658422362 | 0.976028307869 |
|Negative | 0.929024081115 | 0.396430502975 | 0.555724033359 |

Clearly, there was a problem with the performance on Negative examples.

I decided to reduce the dataset by using lemmatization and stopwords.

Additionally, in order to address the issue of the inbalanced dataset I
divided the positive training examples into an arbitrary number between
1 and 8, then pick one of those buckets and merge it with the negative
examples. In other words I undersampled the positive examples and the
number of positive examples used as chosen by cross validation.

Next, I runned a feature selection process to extract the high
information unigrams.

The number of divisions, which bucket, and the number of features were
selected using cross validation simultaneously.
The classifiers were trained with each of these divisions
and evaluated with the validation dataset.

The graphs below show the maximum F measure among all numbers of divisions
and positive bucket selected for a specific number of features. In other
words.
![Max Neg F measure](/neg_f_measure.png?raw=true)
![Max Pos F measure](/pos_f_measure.png?raw=true)
