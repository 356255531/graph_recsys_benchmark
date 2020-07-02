import re
import pandas as pd
from os.path import join
import requests
import json
import tqdm


def parse_ml(dir):
    """
    Read the movielens dataset from .dat file
    :param dir: the path to raw files (users.dat, movies.dat, ratings.dat)
    :param debug: the portion of ratings userd, float
    :return: users, movies, ratings, pandas.DataFrame
    """

    users = []
    with open(join(dir, 'users.dat')) as f:
        for line in f:
            id_, gender, age, occupation, zip_ = line.strip().split('::')
            users.append({
                'uid': int(id_),
                'gender': gender,
                'age': age,
                'occupation': occupation,
                'zip': zip_,
            })
    users = pd.DataFrame(users)

    movies = []
    # parser movies
    with open(join(dir, 'movies.dat'), encoding='latin1') as f:
        for line in f:
            id_, title, genres = line.strip().split('::')
            genres_set = set(genres.split('|'))

            # extract year
            assert re.match(r'.*\([0-9]{4}\)$', title)
            year = title[-5:-1]
            title = title.split(', The')[0].split(' (')[0].split(', A')[0].strip()

            data = {'iid': int(id_), 'title': title, 'year': int(year)}
            for g in genres_set:
                data[g] = True
            movies.append(data)
    movies = (
        pd.DataFrame(movies)
            .fillna(False)
            .astype({'year': 'category'}))

    apikey = ''
    key1 = 'e760129c'
    key2 = 'e44e5305'
    key3 = '8403a97b'
    key4 = '192c6b0e'

    directors_strs = []
    actors_strs = []
    writer_list = []

    pbar = tqdm.tqdm(zip(movies.title, movies.year), total=movies.shape[0])
    for i, (title, year) in enumerate(pbar):
        pbar.set_description('Get item resources')
        if i in range(0, 1000):
            apikey = key1
        if i in range(1000, 2000):
            apikey = key2
        if i in range(2000, 3000):
            apikey = key3
        if i in range(3000, 4000):
            apikey = key4


        try:
            movie_url = "http://www.omdbapi.com/?" + "t=" + title + "&y=" + str(year) + "&apikey=" + apikey
            r = requests.get(movie_url)
            movie_info_dic = json.loads(r.text)
        except:
            try:
                movie_url = "http://www.omdbapi.com/?" + "t=" + title + "&apikey=" + apikey
                r = requests.get(movie_url)
                movie_info_dic = json.loads(r.text)
            except:
                movie_info_dic = dict()

        director = ','.join(movie_info_dic.get('Director', '').split(', '))
        actor = ','.join(movie_info_dic.get('Actors', '').split(', '))
        writer = ','.join([writer.split(' (')[0] for writer in movie_info_dic.get('Writer', '').split(', ')])
        # poster = movie_info_dic.get('Poster', None)

        directors_strs.append(director)
        actors_strs.append(actor)
        writer_list.append(writer)

    movies['directors'] = directors_strs
    movies['actors'] = actors_strs
    movies['writers'] = writer_list

    ratings = []
    with open(join(dir, 'ratings.dat')) as f:
        for line in f:
            user_id, movie_id, rating, timestamp = [int(_) for _ in line.split('::')]
            ratings.append({
                'uid': user_id,
                'iid': movie_id,
                'rating': rating - 1,
                'timestamp': timestamp,
            })
    ratings = pd.DataFrame(ratings)

    return users, movies, ratings
