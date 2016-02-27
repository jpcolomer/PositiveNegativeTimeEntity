# Predict Positive or Negative dialogue act

## Dataset Peculiarities
* The first peculiarity is that the dataset is quite imbalanced. There are ~80.000 examples and only ~7% have NEGATIVE_TIME
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

As a baseline I trained Naive Bayes, SVM L1, SVM L2 classifiers with the training set
and evaluated it on the validation set.

|Naive Bayes | Precision | Recall | F-measure|
|------------|-----------|--------|----------|
|Positive | 0.985006412499 | 0.883151094478 | 0.93130208563 |
|Negative | 0.353424340583 | 0.826122228231 | 0.495057527143 |

|SVM L1 | Precision | Recall | F-measure|
|------------|-----------|--------|----------|
|Positive | 0.953855070147 |  0.997867491794 | 0.975365028763 |
|Negative | 0.931589537223 | 0.375608436993 | 0.535363268452 |

|SVM L2 | Precision | Recall | F-measure|
|------------|-----------|--------|----------|
|Positive | 0.955461208796 | 0.997470259873 | 0.976013910909 |
|Negative | 0.924137931034 | 0.398593834505 | 0.556962025316 |

Clearly, there was a problem with the recall, hence with the F-measure on Negative examples.

I decided to reduce the dataset by using lemmatization and stopwords.

Additionally, in order to address the issue of the imbalanced dataset I
divided the positive training examples into an arbitrary number between
1 and 8, then pick one of those buckets and merge it with the negative
examples. In other words I undersampled the positive examples and the
number of positive examples used as chosen by cross validation.

Next, I runned a feature selection process to extract the high
information unigrams.

The number of divisions, which bucket, and the number of features were
selected using cross validation simultaneously.
The classifiers were trained with each of these divisions
and evaluated using the validation dataset.

The graphs below show the maximum F measure among all numbers of divisions
and positive bucket selected for a specific number of features. In other
words.
![Max Neg F measure](/neg_f_measure.png?raw=true)
![Max Pos F measure](/pos_f_measure.png?raw=true)

The best classifier in terms of negative F measure is the SVM L1 with 6
divisions and using the second bucket and 43.000 features.

Next, I experimented with the different features and found out that by
not using the negation neither removing the stopwords the Negative F
measure increased by 2%.

The next step was crossvalidating the cost parameter of this classifier.
By using C=0.7 it achieves the maximum Negative F-measure.

![Cross validate cost C](/c_cross_val.png?raw=true)

The performance of the tunned model against the validation dataset is:

|SVM L1 | Precision | Recall | F-measure|
|------------|-----------|--------|----------|
|Positive | 0.972502348888 |  0.973803600176 | 0.97315253954 |
|Negative | 0.655200880572 | 0.643861546782 | 0.64948172395 |

Finally, the tunned model was evaluated using the test dataset.

|SVM L1 | Precision | Recall | F-measure|
|------------|-----------|--------|----------|
|Positive | 0.974849094567 |  0.971266288005 | 0.973054393305 |
|Negative | 0.635284139101 | 0.666370106762 | 0.650455927052 |

Although, the F-measure increased ~0.2 points in the validation dataset
and the performance on the test dataset is consistent, it's still not good
enough. I believe I have to use more advanced features in orther to
achieve a better performance.

One simple option that I didn't experiment is using Bigrams.

But I think that to correctly address the problem of multiple time
entities I would have to use a dependency tree and extract information
from the edges of the graph. Unfortunately, I still don't know how to do
this.

Finally, like in sentiment analysis, the use of lexicons in this
problem could also increment the performance.
