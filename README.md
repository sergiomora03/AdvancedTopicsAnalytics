# Advanced Topics in Analytics

*Instructor: Sergio A. Mora Pardo*

- email: <sergioa.mora@javeriana.edu.co>
- github: [sergiomora03](http://github.com/sergiomora03)


Knowledge of the challenges and solutions present in specific situations of organizations that require advanced and special handling of information, such as text mining, process mining, data flow mining (stream data mining) and social network analysis. This module on Natural Language Processing  will explain how to build systems that learn and adapt using real-world applications. Some of the topics to be covered include text preprocessing, text representation, modeling of common NLP problems such as sentiment analysis, similarity, recurrent models, word embeddings, introduction to lenguage generative models. The course will be project-oriented, with emphasis placed on writing software implementations of learning algorithms applied to real-world problems, in particular, language processing, sentiment detection, among others.


## Requiriments 
* [Python](http://www.python.org) version >= 3.7;
* [Numpy](http://www.numpy.org), the core numerical extensions for linear algebra and multidimensional arrays;
* [Scipy](http://www.scipy.org), additional libraries for scientific programming;
* [Matplotlib](http://matplotlib.sf.net), excellent plotting and graphing libraries;
* [IPython](http://ipython.org), with the additional libraries required for the notebook interface.
* [Pandas](http://pandas.pydata.org/), Python version of R dataframe
* [Seaborn](stanford.edu/~mwaskom/software/seaborn/), used mainly for plot styling
* [scikit-learn](http://scikit-learn.org), Machine learning library!

A good, easy to install option that supports Mac, Windows, and Linux, and that has all of these packages (and much more) is the [Anaconda](https://www.continuum.io/).

GIT!! Unfortunatelly out of the scope of this class, but please take a look at these [tutorials](https://help.github.com/articles/good-resources-for-learning-git-and-github/)

## Evaluation

* 50% Project
* 40% Exercises
* 10% Class participation

## Slack Channel
[Join here! <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/Slack_icon_2019.svg/2048px-Slack_icon_2019.svg.png" width="40" height="40" >](https://join.slack.com/t/03065523302654espacio/shared_invite/zt-22271yk06-zKmBDlbdyjmjth7sdj7WnA) 

## Schedule


 ### Intro Natural Language Processing
| Date | Session         | Notebooks/Presentations          | Exercises |
| :----| :----| :------------- | :------------- | 
| August 19th | Introduction to NLP |  <ul><li>[1 - Introduction to NLP](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/Introduction%20to%20NLP.pdf) </li></ul> <ul><li>[2 - NLP Pipeline](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/NLP%20Pipeline.pdf) </li></ul> | <ul><li>[E1 - Tokenization](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L1-Tokenization.ipynb) </li> </ul> | 

### Text Representation
| Date | Session         | Notebooks/Presentations          | Exercises |
| :----| :----| :------------- | :------------- | 
| August 19th | Space Vector Models |  <ul><li>[1 - Basic Vectorizarion Approaches](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/Basic%20Vectorizarion%20Approaches.pdf) </li><li>[L2 - OneHot Encoding](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L1-OneHotEncoding.ipynb) </li><li>[L3 - Bag of Words](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L2-BagOfWords.ipynb) </li><li>[L4 - N-grams](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L3-ngrams.ipynb) </li><li>[L5 - TF-IDF](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L4-TFIDF.ipynb) </li><li>[L6 - Basic Vectorization Approaches](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L5-BasicVectorizationApproaches.ipynb) </li></ul> | <ul><li>[E2 - Sentiment Analysis](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/exercises/E1-SentimentPrediction.ipynb) </li> <li>[P1 - Movie Genre Prediction](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/exercises/P1-MovieGenrePrediction.ipynb) </li></ul> | 
| August 26th | Distributed Representations | <ul><li>[2 - Word Embbedings](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/Word%20Embeddings.pdf)</li><li>[L7 - Text Similarity](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L7-TextSimilarity.ipynb) </li><li> [L8 - Exploring Word Embeddings](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L8-ExploringWordEmbeddings.ipynb)</li> <li>[L9 - Song Embeddings](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L9-SongEmbeddings.ipynb)</li> <li>[L10 - Visualizing Embeddings](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L10-VisualizingEmbeddingsUsingTSNE.ipynb)</li>|  <li>[E2 - Homework Analysis](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/exercises/E2-HomeworksAnalysis.ipynb) </li><li> [E3 - Song Embedding Visualization](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/exercises/E3-SongEmbeddingsVisualization.ipynb)</li> <li>[E4 - Spam Classification](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/exercises/E4-SpamClassification.ipynb)</li>|

### NLP with Deep Learning 
| Date | Session         | Notebooks/Presentations          | Exercises |
| :----| :----| :------------- | :------------- | 
| September 2nd | Deep Learning in NLP (RNN, LSTM, GRU) | <ul><li>[3 - Intro Deep Learning](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/Intro%20Deep%20Learning.pdf)</li><li>[4 - RNN, LSTM, GRU](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/RNN%2C%20LSTM%20and%20GRU.pdf)</li><li>[L11 - Introduction Deep Learining MLP](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L11-IntroductionDeepLearningMLP.ipynb)</li><li>[L12 - Deep Learning with Keras](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L12-DeepLearning_keras.ipynb)</li><li>[L13 - Recurrent Neural Network and LSTM](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L13-RecurrentNeuralNetworks_LSTM.ipynb)</li><li>[L14 - Headline Generator](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L14-Headline_Generator.ipynb)</li></ul> | <ul><li>[E5 - Neural Networks in Keras](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/exercises/E5-NeuralNetworksKeras.ipynb)</li><li>[E6 - RNN, LSTM, GRU](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/exercises/E6-RNN_LSTM_GRU.ipynb)</li></ul>|
| September 9th | Attention, Tranformers and BERT | <ul><li>[5 - Encoder-Decoder](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/Encoder-Decoder.pdf)</li><li>[6 - Attention Mechanisms and Transformers](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/Attention%20Mechanism.pdf)</li><li>[7 - BERT and Family](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/BERT.pdf)</li><li>[L16 - Positional Encoding](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L15-transformer_positional_encoding_graph.ipynb)</li><li>[L17 - BERT for Sentiment Clasification](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L16-BERT_for_sentiment_classification.ipynb)</li><li>[L18 - Transformers Introduction](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L17-TransformersIntroduction.ipynb)</li></ul> | <ul><li>[E7 - Text Summary](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/exercises/E7-TextSummary.ipynb)</li><li>[E8 - Question Answering](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/exercises/E8-QuestionAnswer.ipynb)</li><li>[E9 - Open AI](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/exercises/E9-OpenAI.ipynb)</li></ul> |

### Intro Graph
| Date | Session         | Notebooks/Presentations          | Exercises |
| :----| :----| :------------- | :------------- | 
|September 16th | Week of Reflection - No class | No class | No class |
|September 23th | Intro to Graphs | <ul><li>[L19 - Intro to Graphs](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L18_IntroductionGraphs.ipynb)</li></ul> | |
|September 23th | Graphs Metrics | <ul><li>[L20 - Graph Metrics](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L19-GraphMetrics.ipynb)</li> <li>[L21 - Graphs Benchmarks](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L20-GraphsBenchmarks.ipynb)</li> <li>[L22 - Facebook Analysis](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L21-FacebookNetworkAnalysis.ipynb)</li> </ul> | [E10 - Twitter Analysis](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/exercises/E10-TwitterNetworkAnalysis.ipynb) |

### Graph Representation Learning
| Date | Session         | Notebooks/Presentations          | Exercises |
| :----| :----| :------------- | :------------- | 
|october 7th | Adjacency Matrix | | |
|october 7th | Node Embedding (Node2Vec) | [L23 - Graph Embedding](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L22-GraphEmbedding.ipynb) | |

### Machine Learning with Graphs
| Date | Session         | Notebooks/Presentations          | Exercises |
| :----| :----| :------------- | :------------- | 
|october 14th | Graph Neural Network | | |
|october 21th | Social Network Analysis | | |

## Interest Links 🔗
Module | Topic | Material |
| :----| :----| :----|
| NLP | Word Embedding Projector |[Tensorflow Embeddings Projector](https://projector.tensorflow.org/)|
| NLP | Time Series with LSTM | [ARIMA-SARIMA-Prophet-LSTM](https://www.kaggle.com/code/sergiomora823/time-series-analysis-arima-sarima-prophet-lstm) |
| NLP | Stanford | [Natural Language Processing with Deep Learning](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/)
| GML | Stanford | [CS224W: Machine Learning with Graphs](http://web.stanford.edu/class/cs224w/)

## Extra Material
Module | Topic | Material |
| :----| :----| :----|
| NLP | Polarity | [Sentiment Analysis - Polarity](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/X1-SentimentAnalysisPolarity.ipynb) |
| NLP | Image & Text | [Image Captions](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/X2-image_captions.ipynb) |
| ML | Hyperparameter Tuning [WIP] | <ul><li>[Exhaustive Grid Search]()</li> <li>[Randomized Parameter Optimization]()</li> <li>[Automate Hyperparameter Search]()</li></ul> |
| NLP | Neural Style Transfer | [Style Transfer](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/X3-style_transfer.ipynb) |



