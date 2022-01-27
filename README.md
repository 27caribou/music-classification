# Music Classification using a Convolutional Neural Network


## Motivations

As I started diving into Deep Learning, I thought it would be a good idea to work on a relatable project to reinforce what I learned so far. That is how I came across various articles talking about how you could use Deep Learning to classify songs. Being a very active Spotify user, I got excited about the premise.

During my research, I regularly came across two ways of approaching the problem: **Content-based filtering** and **Collaborative filtering**.

Spotify, for example, provided most of its recommendations based on collaborative filtering, using historical usage data. This concept is based on the assumption that *if user A and user B listen to almost the same songs, then their tastes are similar*. Conversely, it means the songs sound similar. This model makes it very flexible and can be applied in other fields, such as books, movies, etc. However, the main issue was that it favored more popular songs and was harder to introduce new songs (cold start).

Another approach involved content-based filtering which then included analyzing the item itself i.e. determining which features were the most important in making a judgement. However in this case, data collection can be more complicated, and you have to decide on whether to use metadata or the audio content.

The concept of content-based filtering appealed more to me, so I looked for a dataset that would fit that approach.

## About the Project

The dataset used in this project is the [GTZAN Dataset](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification).

The basic idea of this project is to classify songs into 10 genres of music by passing their **MFCCs** through a convolutional neural network. Mel Frequency Cepstral Coefficients (MFCCs) are a small set of features (usually about 10-20) that concisely describe the overall shape of the sound. Just like mel spectrograms, MFCCs have also proven to provide good results in music AI projects.

The MFCCs are extracted from every song found in the genres_original folder. Due to the limited number of samples given (100 songs * 10 genres), I used different augmentation techniques to produce more data to feed into the CNN.

**As a result, I was able to produce a 97% accuracy on my training data and 94% accuracy on my validation data, which is better than the 90% benchmark I set using the ANN produced from the data in the features_3_sec.csv file**.

### 1_Features_3_sec_model

A features_3_sec.csv file comes with the dataset, and it reports the mean and variance of various audio features such as spectral centroid and zero crossing obtained after slicing all the songs into 3s segments. I created an Artificial Neural Network with the data obtained from this file, and **used its results as a benchmark for my own model**. I ran the data through an ANN to see how good the data was, and **was able to deliver an accuracy of 90-92% with this data**.

![model_training_from_csv](https://github.com/27caribou/music-classification/blob/main/Snapshots/loss_and_accuracy_features.png)
![confusion_matrix_from_csv](https://github.com/27caribou/music-classification/blob/main/Snapshots/matrix_features.png)
![classification_report_from_csv](https://github.com/27caribou/music-classification/blob/main/Snapshots/classification_report_features.png)

### Preprocessing

Most of the preprocessing has to do with extracting MFCCs from audio signals. I also sliced every audio sample into 3s segment, since the features_3_sec.csv file proved that we could get good results with just 3 seconds of content. Despite all my efforts on changing the structure of my CNN model, my accuracy was not very good. 

![model_training_no_augmentation](https://github.com/27caribou/music-classification/blob/main/Snapshots/loss_and_accuracy_no_augmentation.png)

Instead of focusing of my model, I decided to take a more data-centric approach. I did some research and found useful audio transformations relevant to this project: *pitch scaling, random gain and white noise*. That way, I'm not only increasing the amount of data, but I'm also making the model more robust to situations when the audio is not as clean.

Original MFCC
![model_training_no_augmentation](https://github.com/27caribou/music-classification/blob/main/Snapshots/sample_mfcc_original.png)
Pitch scale applied
![model_training_no_augmentation](https://github.com/27caribou/music-classification/blob/main/Snapshots/sample_mfcc_pitch_scale.png)
Random gain applied
![model_training_no_augmentation](https://github.com/27caribou/music-classification/blob/main/Snapshots/sample_mfcc_random_gain.png)
White noise applied
![model_training_no_augmentation](https://github.com/27caribou/music-classification/blob/main/Snapshots/sample_mfcc_white_noise.png)



### Neural Network

![cnn_architecture](https://github.com/27caribou/music-classification/blob/main/Snapshots/neural_architecture.png)

## Training Results

```
Training Accuracy = 97.02%
Validation Accuracy = 94.19 %
```
![model_training_augmented](https://github.com/27caribou/music-classification/blob/main/Snapshots/loss_and_accuracy_augmented.png)
![classification_report_augmented](https://github.com/27caribou/music-classification/blob/main/Snapshots/classification_report_augmented.png)
![confusion_matrix_augmented](https://github.com/27caribou/music-classification/blob/main/Snapshots/matrix_augmented.png)


## Acknowledgements

* Project is mainly inspired by **Vikram Shenoy's** [*work*](https://github.com/VikramShenoy97/Music-Recommendation-Using-Deep-Learning).
* **Valerio Velardo's** [*Youtube channel*](https://www.youtube.com/c/ValerioVelardoTheSoundofAI) is the most useful resource for learning about music in AI (great tutorials!)
* [*Work w/ Audio Data: Visualise, Classify, Recommend*](https://github.com/VikramShenoy97/Music-Recommendation-Using-Deep-Learning) by **Andrada Olteanu**.
* **Ketan Doshi's** articles on [*Audio Deep Learning*](https://towardsdatascience.com/audio-deep-learning-made-simple-part-1-state-of-the-art-techniques-da1d3dff2504)

