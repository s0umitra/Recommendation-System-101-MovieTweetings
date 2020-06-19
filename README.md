# Recommendation-System-101-MovieTweetings

[![forthebadge](https://forthebadge.com/images/badges/60-percent-of-the-time-works-every-time.svg)](https://forthebadge.com)

A Movie Recommendation-System based on 
[MovieTweetings](https://github.com/sidooms/MovieTweetings/tree/master/latest) Dataset

This Project is a part of Data Science Nanodegree Program by Udacity in collaboration with IBM. The initial dataset contains the users, movies, and ratings of the movies. The aim of the project is to build a Recommendation System that recommend movies in the best possible way.

This system uses Hybrid method to make recommendations (the mixed usage of Knowledge based, Collaborative and Content Based Filtering).

### Execution Flow:

```
rec.fit()
rec.make_recommendations()
predict_rating()
```

### Sample Output:

```
# _id = 66
# id_type = 'user'

# *********************************************************************************************

	Top Recommendations for user 66 are:
   Movie ID                              Movie Name
0    454876                       Life of Pi (2012)
1   1024648                             Argo (2012)
2   1853728            Jack the Giant Slayer (2013)
3   1659337  The Perks of Being a Wallflower (2012)
4   1351685                 Django Unchained (2012)

For user 66 we predict a 11.49 rating for the movie     Life of Pi (2012).
For user 66 we predict a 10.65 rating for the movie     Argo (2012).
For user 66 we predict a 10.63 rating for the movie     Django Unchained (2012).
For user 66 we predict a 10.49 rating for the movie     The Perks of Being a Wallflower (2012).
For user 66 we predict a 10.26 rating for the movie     Jack the Giant Slayer (2013).

```

### License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](https://github.com/s0umitra/Recommendation-System-101-MovieTweetings/blob/master/LICENSE)

This software is licenced under [MIT](https://github.com/s0umitra/Recommendation-System-101-MovieTweetings/blob/master/LICENSE)

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
