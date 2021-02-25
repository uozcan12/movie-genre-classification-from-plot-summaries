# Predicting Movie Genres Based on Plot Summaries

**Summary :** Movie plot summaries are expected to reflect the genre of movies since many spectators read the plot summaries before deciding to watch a movie. Movie plot summaries reflect the genre of the movies such as action, drama, horror, etc., such that people can easily capture the genre information of the movies from their plot summaries. Especially, several sentences in the plot summaries are high representatives of genre of the movie. People usually read the plot summaries of movies before watching them to get an idea about the movie. Therefore, plot summaries are written in such a way that they convey the genre information to the people. In this project, we will predict movie genres based on plot summaries and will compare different algorithms.

**Goal :** This project explores NLP and LSTM methods and several Machine Learning methods to predict movie genres based on plot summaries.
Data Description: There are 3 databases for now. MovieLens[1] database , MPST: Movie Plot Synopses and CMU Movie Summary Corpus dataset. In addition, we can work on IMDB dataset according to "Predicting Movie Genres Based on Plot Summaries" article[2]. 
MPST: Movie Plot Synopses(https://www.kaggle.com/cryptexcode/mpst-movie-plot-synopses-with-tags?select=mpst_full_data.csv):
    • imdb_id
    • title
    • plot_synopsis
    • tags
    • split
    • synopsis_source

**Algorithm: ** The first techniques that come to mind are NLP techniques and LSTM. But depending on the progress of the project, different algorithms can be used.
**Future Work:** If we reached good accuracy on movie genres, we can predict movie ratings and movie actors/actress/director etc.
Inspiration
    • Content-Based Movie Recommender:
Recommend movies with plots similar to those that a user has rated highly.
    • Movie Plot Generator:
Generate a movie plot description based on seed input, such as director and genre
    • Information Retrieval: 
Return a movie title based on an input plot description

**Literature Survey: **
[1] Ertugrul, Ali Mert & KARAGOZ, Pinar. (2018). Movie Genre Classification from Plot Summaries Using Bidirectional LSTM. 10.1109/ICSC.2018.00043. 
[2] Hoang, Quan. (2018). Predicting Movie Genres Based on Plot Summaries.

