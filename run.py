from recommender import Recommender
import pandas as pd


def run(rec_type):
    if rec_type == 'user':

        movie_ids, movie_names = rec.make_recommendations(_id, id_type)
        rec_movies = {'Movie ID': movie_ids,
                      'Movie Name': movie_names
                      }

        print('\nTop Recommendations for user {} are:'.format(_id))
        print(pd.DataFrame(data=rec_movies), end='\n\n')

        for i in movie_ids:
            # predict
            rec.predict_rating(user_id=_id, movie_id=i)

    elif rec_type == 'movie':

        movie_names = rec.make_recommendations(_id, id_type)

        print('\nTop Recommendations for movie id {} are:'.format(_id))
        print(movie_names, end='\n\n')


if __name__ == '__main__':
    # instantiate recommender
    rec = Recommender()

    # fit recommender
    rec.fit(reviews_pth='data/train_data.csv', movies_pth='data/movies_clean.csv', learning_rate=.01, iterators=1)

    # make recommendations
    # 'user' or 'movie'

    _id = 66
    id_type = 'user'

    """
    _id = 1675434
    id_type = 'movie'"""

    run(id_type)

    ''' 
    More Examples:
    print(rec.make_recommendations(8, 'user'))  # user in the dataset
    print(rec.make_recommendations(1, 'user'))  # user not in dataset
    print(rec.make_recommendations(1853728))  # movie in the dataset
    print(rec.make_recommendations(1))  # movie not in dataset
    print(rec.n_users)
    print(rec.n_movies)
    print(rec.num_ratings)
    '''
