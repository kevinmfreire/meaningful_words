# Sentiment Analysis

## Table of Content
* Overview
* Backgraound and Motivation
* Goals
* Datasets
* Practical Applications
* Usage
* Milestones
* References

## Overview

As Social Media start to influence the life of many children, teenagers, adults and now as we all know grandparents it can have a big impact on mental health.  Those who have instagram, twitter, facebook, and tiktok hace been exposed to various posts that has affected us emotionally and psychologically.  It can go from viewing a friends posts on travelling to different countries while your sitting at home doing work, if can be an invite that you never received, or any post that got you thinking "Their life is so much better than mine".  Social media has also been a medium for spreading negativity, such as fake news, toxic comments, or even to ruin someones reputation by spreading rumours. However, it can't all be that bad? After all social media has allowed people to start their own business, create their own mood board, maybe even share their self-imporvement journey.  What if social media can move in the direction of elevating one another?

## Background and Motivation

* Social media has influenced some people to think that their life is not interesting, they aren't as happy as they should be, they should be getting more "likes", have more followeres because if they don't then they are considered losers.  All this develops anxiety and stress that can lead someone to become angry, depressed or feel extremly lonely.
* The rise on mental health for various social media users.

## Goals

* Build a classifier to distinguish comments, posts, texts as negative, positive or neutral.
* Build a recommender system to feed into the individuals newsfeed positive posts such as images, inspirational quotes that elevates their mood.

## Datasets
* [Kaggle Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) - Good but not a lot of labels
* [Kaggle Twitter Tweets Sentiment Dataset](https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset) - Good but not a lot of labels
* [OpenML Emotions--Sensor-Data-Set](https://www.openml.org/search?type=data&status=active&id=43756) - This one is not bad
* [Sentiment Analysis in Text](https://data.world/crowdflower/sentiment-analysis-in-text) - I like this one
* [Emotions Dataset](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp) - I like this one

## Practical Applications
* Social media apps tracking users in the background for mental health issues.
* Suggest accounts to follow that can feed the users newsfeed with positive ideas to prevent user from having negative thoughts that can cause them to doubt, or talk negatively about themselves.
* Integrated as a web extension to track users if they are experiencing any mental health issues.

## Usage
**Clone Repo:** `git clone https://github.com/kevinmfreire/sentiment-analysis.git`

**Setup a virtual environment:** `virtualenv .virtualenv/sentiment-analysis`

**Activate virtual environment:** `source .virtualenv/sentiment-analysis`

**Install all requirements using pip or any other package installer:** `pip install -r requirements.txt`

**get up the required NLTK packages** `python nltk_setup.py`

## Milestones
### Phase 1:
Develope a webapp  where the user inputs their sentence (email, tweet) we would return the label of that text (negative, positive, neutral sentiment)

### Phase 2:
Talk to the Twitter API to gather the users latest tweets and analyze them (send them to the model) and return the users overall sentiment (mood).

### Phase 3:
Include recommender system feeds the users to follow accounts that can imporve someones mental health.

## References
* [Text Classification Using Logistic Regression](https://medium.com/analytics-vidhya/applying-text-classification-using-logistic-regression-a-comparison-between-bow-and-tf-idf-1f1ed1b83640)
* [Text Classification using SVM and Naive Bayes](https://medium.com/@bedigunjit/simple-guide-to-text-classification-nlp-using-svm-and-naive-bayes-with-python-421db3a72d34)
* [Text Classification Using K-Nearest Neighbor](https://medium.com/@ashins1997/text-classification-456513e18893)
