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

## Deadlines
| Session | Activity | Deadline | Comments |
| :---- | :----| :---- | :---- | 
| Deep Learning | <ul>Exercises</ul> <ul>Project</ul>| <ul>March 21th</ul> | Expo March 22th |
| NLP | <ul>Exercises</ul> <ul>Project</ul>| <ul>April 25th</ul> <ul>April 11th</ul> | Expo April 12th |
| Graph Learning | <ul>Exercises</ul> <ul>Project</ul>| <ul>May 24th</ul> |  |
| Final grade | <ul>project</ul> | <ul>May 31thth</ul> |  |


## Slack Channel
[Join here! <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/Slack_icon_2019.svg/2048px-Slack_icon_2019.svg.png" width="40" height="40" >](https://join.slack.com/t/advancedtopic-ifp1252/shared_invite/zt-31dvj2p0j-N8tjsMZkyZWC70iENzgy~g)

## Schedule

### Basic Methods MLOps
| Date | Session         | Notebooks/Presentations          | Exercises |
| :----| :----| :------------- | :------------- | 
| March 1st | Machine Learning Operations (MLOps) | <ul><li> [Intro MLOps](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/MLOps.pdf) </li> </ul> | |
| March 1st | ML monitoring & Data Drift | <ul><li> [Intro Data Drift](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/IntroDataDrift.ipynb) <ul><li>[L2 - Intro Data Drift](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/default_stattest_adult.ipynb)</li></ul><ul><li>[L3 - Intro Model Monitoring](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/ibm_hr_attrition_model_validation.ipynb) </li></ul>  | <ul><li>[E1 - Data Drift in Used Vehicle Price Prediction](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/exercises/E2-UsedVehiclePricePredictionDrift.ipynb) </li> </ul> | 
| March 1st | Machine Learning as a Service (AIaaS) |  <ul><li>[1 - Intro to APIs](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/IntroductionToAPIs.ipynb)  <ul><li>[L1 - Model Deployment](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/Model_Deployment.ipynb) </li></ul> | <ul><li>[E2 - Model Deployment in Used Vehicle Price Prediction](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/exercises/E1-UsedVehiclePricePredictionDeployment.ipynb) </li> </ul> | 

### Intro Deep Learning
| Date | Session         | Notebooks/Presentations          | Exercises |
| :----| :----| :------------- | :------------- | 
| March 8th | First steps in deep learning |  <ul><li>[3 - Intro Deep Learning](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/Intro%20Deep%20Learning.pdf)</li><li>[L3 - Introduction Deep Learining MLP](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L11-IntroductionDeepLearningMLP.ipynb)</li><li>[L4 - Simple Neural Network (handcraft)](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/SimpleNeuralNetwork.ipynb)</li><li>[L5 - Simple Neural Network (Images)](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/SimpleNeuralNetworkImage.ipynb)</li><li>[L6 - Deep Learning with Keras](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/DeepLearningTensorflow.ipynb)</li><li>[L7 - Deep Learning with Pytorch](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/DeepLearningPyTorch.ipynb)</li></ul> | <ul><li>[E3 - Neural Networks in Keras and PyTorch](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/exercises/E5-NeuralNetworksKeras.ipynb)</li></ul> | 
| March 15th | Deep Computer Vision |  <ul><li>[4 - Convolutional Neural Networks](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/Deep%20Computer%20Vision.pdf) <li>[L5 - CNN with TensorFlow](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/CNN-TensorFlow.ipynb) </li> <li>[L6 - CNN with PyTorch 🔥](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/CNN-PyTorch.ipynb)</li><li>[L7 - Tranfer Learning with TensorFlow](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/PretrainedModelsTensorFlow.ipynb) </li></ul>  | <ul> <li>[E4 - Tranfer Learning with PyTorch](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/exercises/E4-PretrainedModelsPytorch.ipynb) </li> </ul> | 
| March 22th | Computer Vision Project | Exercises Deadline | [P1 - Frailejon Detection (a.k.a "Big Monks Detection")](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/exercises/P0_BigMonksDetection.ipynb) |

### Intro Natural Language Processing
| Date | Session         | Notebooks/Presentations          | Exercises |
| :----| :----| :------------- | :------------- | 
| March 22th | Introduction to NLP |  <ul><li>[1 - Introduction to NLP](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/Introduction%20to%20NLP.pdf) </li></ul> <ul><li>[2 - NLP Pipeline](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/NLP%20Pipeline.pdf) </li><li>[E1 - Tokenization](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L1-Tokenization.ipynb)</li></ul> | | 

### Text Representation
| Date | Session         | Notebooks/Presentations          | Exercises |
| :----| :----| :------------- | :------------- | 
| March 22th | Space Vector Models |  <ul><li>[1 - Basic Vectorizarion Approaches](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/Basic%20Vectorizarion%20Approaches.pdf) </li><li>[L2 - OneHot Encoding](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L1-OneHotEncoding.ipynb) </li><li>[L3 - Bag of Words](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L2-BagOfWords.ipynb) </li><li>[L4 - N-grams](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L3-ngrams.ipynb) </li><li>[L5 - TF-IDF](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L4-TFIDF.ipynb) </li><li>[L6 - Basic Vectorization Approaches](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L5-BasicVectorizationApproaches.ipynb) </li></ul> | <ul><li>[E2 - Sentiment Analysis](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/exercises/E1-SentimentPrediction.ipynb) </li> </ul> | 
| March 29th | Distributed Representations | <ul><li>[2 - Word Embbedings](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/Word%20Embeddings.pdf)</li><li>[L7 - Text Similarity](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L7-TextSimilarity.ipynb) </li><li> [L8 - Exploring Word Embeddings](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L8-ExploringWordEmbeddings.ipynb)</li> <li>[L9 - Song Embeddings](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L9-SongEmbeddings.ipynb)</li> <li>[L10 - Visualizing Embeddings](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L10-VisualizingEmbeddingsUsingTSNE.ipynb)</li>|  <li>[E2 - Homework Analysis (Bonus)](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/exercises/E2-HomeworksAnalysis.ipynb) </li><li> [E3 - Song Embedding Visualization](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/exercises/E3-SongEmbeddingsVisualization.ipynb)</li> <li>[E4 - Spam Classification](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/exercises/E4-SpamClassification.ipynb)</li>|

### NLP with Deep Learning 
| Date | Session         | Notebooks/Presentations          | Exercises |
| :----| :----| :------------- | :------------- | 
| April 5th | Deep Learning in NLP (RNN, LSTM, GRU) | <ul><li>[4 - RNN, LSTM, GRU](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/RNN%2C%20LSTM%20and%20GRU.pdf)</li><li>L12 - NLP with Keras</li><li>[L11 - NLP with Keras](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L12-DeepLearning_keras_NLP.ipynb)</li><li>[L13 - Recurrent Neural Network and LSTM](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L13-RecurrentNeuralNetworks_LSTM.ipynb)</li><li>[L14 - Headline Generator](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L14-Headline_Generator.ipynb)</li></ul> | <ul><li>[E5 - Neural Networks in Keras for NLP](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/exercises/E5-NeuralNetworksKerasNLP.ipynb)</li><li>[E6 - Neural Networks in PyTorch for NLP](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/exercises/E5-NeuralNetworksPyTorchNLP.ipynb)</li><li>[E7 - RNN, LSTM, GRU](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/exercises/E6-RNN_LSTM_GRU.ipynb)</li></ul>|
| April 12th | NLP Project | | [P1 - Movie Genre Prediction](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/exercises/P1-MovieGenrePrediction.ipynb) |
| April 12th | Attention, Tranformers and BERT | <ul><li>[5 - Encoder-Decoder](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/Encoder-Decoder.pdf)</li><li>[6 - Attention Mechanisms and Transformers](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/Attention%20Mechanism.pdf)</li><li>[7 - BERT and Family](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/BERT.pdf)</li><li>[L16 - Positional Encoding](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L15-transformer_positional_encoding_graph.ipynb)</li><li>[L17 - BERT for Sentiment Clasification](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L16-BERT_for_sentiment_classification.ipynb)</li><li>[L18 - Transformers Introduction](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L17-TransformersIntroduction.ipynb)</li></ul> | <ul><li>[E8 - Text Summary](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/exercises/E7-TextSummary.ipynb)</li><li>[E9 - Question Answering](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/exercises/E8-QuestionAnswer.ipynb)</li><li>[E10 - Open AI](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/exercises/E9-OpenAI.ipynb)</li></ul> |
| April 19th | _Holy Week_ | _Holy Week_ | _Holy Week_ | 
| April 25th | | Exercises Deadline | |

### Intro Graph
| Date | Session         | Notebooks/Presentations          | Exercises |
| :----| :----| :------------- | :------------- | 
| April 26th | Intro to Graphs | <ul><li>[Intro to Graphs](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/introGRaphs.pdf)</li><li>[L19 - Intro to Graphs](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L18-IntroductionGraphs.ipynb)</li></ul> | |
| April 26th | Graphs Metrics | <ul><li>[L20 - Graph Metrics](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L19-GraphMetrics.ipynb)</li> <li>[L21 - Graphs Benchmarks](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L20-GraphsBenchmarks.ipynb)</li> <li>[L22 - Facebook Analysis](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L21-FacebookNetworkAnalysis.ipynb)</li> </ul> | [E10 - Twitter Analysis](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/exercises/E10-TwitterNetworkAnalysis.ipynb) |

### Graph Representation Learning
| Date | Session         | Notebooks/Presentations          | Exercises |
| :----| :----| :------------- | :------------- | 
| May 3th | Graph Representation | <ul><li>[Graph Representations](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/GraphRepresentation.pdf)</li><li>[L23 - Graph Embedding](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L22-GraphEmbedding.ipynb)</li><li>[L24 - Deep Walk](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L23-DeepWalk.ipynb)</li><li>[L25 - Node2Vec](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L24-Node2Vec.ipynb)</li><li>[L26 - Recommendation System with Node2Vec](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L25-Node2Vec-RecSys.ipynb)</li></ul> | <li>[E11 - Patent Citation Network (Node2Vec with RecSys)](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/exercises/E11-PatentCitationNetwork.ipynb)</li> |

### Intro to Geometric Deep Learning
| Date | Session         | Notebooks/Presentations          | Exercises |
| :----| :----| :------------- | :------------- | 
| May 10th | Graph Neural Network | <ul><li>[L27 - Graph Neural Networks - Node Features](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L27-GraphNeuralNetworks-NodeFeatures.ipynb)</li><li>[L28 - Pytorch intro with Node2Vec](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L26-Node2Vec-Pytorch.ipynb)</li><li>[L29 - Graph Neural Networks - Adjacency Matrix](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L27-GraphNeuralNetworks-AdjacencyMatrix.ipynb)</li><li>[L31 - Graph Convolutional Networks - Node Classification](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L29-GraphConvolutionalNetworks-NodeClassification.ipynb)</li><li>[L34 - Graph Attention Networks](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L32-GraphAttentionNetworks.ipynb)</li><li>[L33 - Graph Convolutional Networks - Node Regression](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L31-GraphConvolutionalNetworks-NodeRegression.ipynb)</li></ul> | <ul><li>[L30 - Graph Neural Networks - Facebook Page-Page dataset](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L28-GraphNeuralNetworks.ipynb)</li><li>[L32 - Graph Convolutional Networks - Facebook Page-Page dataset](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L30-GraphConvolutionalNetworks-NodeClassification.ipynb)</li><li>[L34 - Graph Attention Networks - Cite Seer](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L33-GraphAttentionNetworks-CiteSeer.ipynb)</li></ul> |
| May 17th | Graph Machine Learning Task [Optional] | <ul><li>[L35 - Graph AutoEncoder - Link Prediction](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L35-LinkPredictionGraphAutoencoder.ipynb)</li><li>[L36 - Graph Variational AutoEncoder - Link Prediction \[extra\]](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L37-LinkPredictionVariationalAutoEncoder.ipynb)</li><li>[L37 - Node2Vec - Link Classification](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L36-LabelClassificationNode2Vec.ipynb)</li><li>[L38 - Graph Isomorphism Network - Graph Classification](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/L34-GraphClassificationGraphIsomorphismNetwork.ipynb)</li></ul> | |
| May 24th | Geometric Deep Learning Project | Exercises Deadline | [P3 - Graph Machine Learning](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/exercises/P2-GraphMachineLearning.pdf) / [P3 - Graph Machine Learning [old < 2022]](https://github.com/sergiomora03/AdvancedTopicsAnalytics/blob/main/notebooks/Proyecto_GML.pdf) |

### Grades
| Date | Session         | Notebooks/Presentations          | Exercises |
| :----| :----| :------------- | :------------- | 
| May 31th | Final Grades | | |

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



