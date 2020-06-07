import numpy as np
import pandas as pd
import lib


class Recommender:
    """
    This Recommender uses FunkSVD to make predictions of exact ratings and uses either FunkSVD or a Knowledge Based
    recommendation (highest ranked) to make recommendations for users. Finally, if given a movie, the recommender
    will provide movies that are most similar as a Content Based Recommender.
    """

    def __init__(self):
        """
        no required attributes to initiate
        """
        self.movies = 0
        self.reviews = 0
        self.user_item_df = 0
        self.user_item_mat = 0
        self.latent_features = 0
        self.learning_rate = 0
        self.iterators = 0
        self.n_users = 0
        self.n_movies = 0
        self.num_ratings = 0
        self.user_ids_series = 0
        self.movie_ids_series = 0
        self.user_mat = 0
        self.movie_mat = 0
        self.ranked_movies = 0

    def fit(self, reviews_pth, movies_pth, latent_features=12, learning_rate=0.0001, iterators=100):
        """This function performs matrix factorization using a basic form of FunkSVD with no regularization
        :param reviews_pth: path to csv with at least the four columns: 'user_id', 'movie_id', 'rating', 'timestamp'
        :param movies_pth: path to csv with each movie and movie information in each row
        :param latent_features: (int) the number of latent features used
        :param learning_rate: (float) the learning rate
        :param iterators: (int) the number of iterations
        :return: None
        """
        # Store inputs as attributes
        self.reviews = pd.read_csv(reviews_pth)
        self.movies = pd.read_csv(movies_pth)

        # Create user-item matrix
        user_vs_item = self.reviews[['user_id', 'movie_id', 'rating', 'timestamp']]
        self.user_item_df = user_vs_item.groupby(['user_id', 'movie_id'])['rating'].max().unstack()
        self.user_item_mat = np.array(self.user_item_df)

        # Store more inputs
        self.latent_features = latent_features
        self.learning_rate = learning_rate
        self.iterators = iterators

        # Set up useful values to be used through the rest of the function
        self.n_users = self.user_item_mat.shape[0]
        self.n_movies = self.user_item_mat.shape[1]
        self.num_ratings = np.count_nonzero(~np.isnan(self.user_item_mat))
        self.user_ids_series = np.array(self.user_item_df.index)
        self.movie_ids_series = np.array(self.user_item_df.columns)

        # initialize the user and movie matrices with random values
        user_mat = np.random.rand(self.n_users, self.latent_features)
        movie_mat = np.random.rand(self.latent_features, self.n_movies)

        # keep track of iteration and MSE
        print("Optimization Statistics")
        print("Iterations | Mean Squared Error ")

        # for each iteration
        for iteration in range(self.iterators):

            # update our sse
            sse_accum = 0

            # For each user-movie pair
            for i in range(self.n_users):
                for j in range(self.n_movies):

                    # if the rating exists
                    if self.user_item_mat[i, j] > 0:

                        # compute the error as the actual minus the dot product of the user and movie latent features
                        diff = self.user_item_mat[i, j] - np.dot(user_mat[i, :], movie_mat[:, j])

                        # Keep track of the sum of squared errors for the matrix
                        sse_accum += diff ** 2

                        # update the values in each matrix in the direction of the gradient
                        for k in range(self.latent_features):
                            user_mat[i, k] += self.learning_rate * (2 * diff * movie_mat[k, j])
                            movie_mat[k, j] += self.learning_rate * (2 * diff * user_mat[i, k])

            # print results
            print("%d \t\t %f" % (iteration + 1, sse_accum / self.num_ratings))

        # SVD based fit
        # Keep user_mat and movie_mat for safe keeping
        self.user_mat = user_mat
        self.movie_mat = movie_mat

        # Knowledge based fit
        self.ranked_movies = lib.create_ranked_df(self.movies, self.reviews)

    def predict_rating(self, user_id, movie_id):
        """
        :param user_id: the user_id from the reviews df
        :param movie_id: the movie_id according the movies df
        :return pred: the predicted rating for user_id-movie_id according to FunkSVD
        """

        try:
            # User row and Movie Column
            user_row = np.where(self.user_ids_series == user_id)[0][0]
            movie_col = np.where(self.movie_ids_series == movie_id)[0][0]

            # Take dot product of that row and column in U and V to make prediction
            pred = np.dot(self.user_mat[user_row, :], self.movie_mat[:, movie_col])

            movie_name = str(self.movies[self.movies['movie_id'] == movie_id]['movie'])[5:]
            movie_name = movie_name.replace('\nName: movie, dtype: object', '')

            print("For user {} we predict a {} rating for the movie {}.".format(user_id, round(float(pred), 2),
                                                                                str(movie_name)))

            return pred

        except():
            print("I'm sorry, but a prediction cannot be made for this user-movie pair. It looks like one of these"
                  "items does not exist in our current database.")

            return None

    def make_recommendations(self, _id, _id_type='movie', rec_num=5):
        """
        :param _id: (int) either a user or movie id
        :param _id_type: (str) "movie" or "user"
        :param rec_num: (int) number of recommendations to return
        :return recs: (array) a list or numpy array of recommended movies like the
                       given movie, or recs for a user_id given
        """
        # if the user is available from the matrix factorization data,
        # I will use this and rank movies based on the predicted values
        # For use with user indexing
        rec_ids, rec_names = None, None
        if _id_type == 'user':
            if _id in self.user_ids_series:
                # Get the index of which row the user is in for use in U matrix
                idx = np.where(self.user_ids_series == _id)[0][0]

                # take the dot product of that row and the V matrix
                preds = np.dot(self.user_mat[idx, :], self.movie_mat)

                # pull the top movies according to the prediction
                indices = preds.argsort()[-rec_num:][::-1]  # indices
                rec_ids = self.movie_ids_series[indices]
                rec_names = lib.get_movie_names(rec_ids, self.movies)

            else:
                # if we don't have this user, give just top ratings back
                rec_names = lib.popular_recommendations(rec_num, self.ranked_movies)
                print("Because this user wasn't in our database,"
                      "we are giving back the top movie recommendations for all users.")

        # Find similar movies if it is a movie that is passed
        else:
            if _id in self.movie_ids_series:
                rec_names = list(lib.find_similar_movies(_id, self.movies))[:rec_num]
            else:
                print("That movie doesn't exist in our database.  Sorry, we don't have any recommendations for you.")

        return rec_ids, rec_names
