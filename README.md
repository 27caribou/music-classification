# Music Classification using a Convolutional Neural Network


## Motivations

As I started diving into Deep Learning, I thought it would be a good idea to work on a relatable project to reinforce what I learned so far. That is how I came across various articles talking about how you could use Deep Learning to classify songs. Being a very active Spotify user, I got excited about the premise.

During my research, I regularly came across two ways of approaching the problem: **Content-based filtering** and **Collaborative filtering**.

Spotify, for example, provided most of its recommendations based on collaborative filtering, using historical usage data. This concept is based on the assumption that *if user A and user B listen to almost the same songs, then their tastes are similar*. Conversely, it means the songs sound similar. This model makes it very flexible and can be applied in other fields, such as books, movies, etc. However, the main issue was that it favored more popular songs and was harder to introduce new songs (cold start).

Another approach involved content-based filtering which then included analyzing the item itself i.e. determining which features were the most important in making a judgement. However in this case, data collection can be more complicated, and you have to decide on whether to use metadata or the audio content.

The concept of content-based filtering appealed more to me, so I looked for a dataset that would fit that approach.

### About the Project

The dataset used in this project is the [GTZAN Dataset](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification).

