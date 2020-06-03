import torch
from dateutil.parser import parser
from torch.utils.data import DataLoader
from os.path import join
from os.path import isfile
import numpy as np
import pandas as pd
import itertools
from collections import Counter
import tqdm
import pickle

from .dataset import Dataset
from torch_geometric.data import download_url, extract_zip
from ..parser import parse_ml


def save_df(df, path):
    df.to_csv(path, sep=';', index=False)


def reindex_df(users, items, interactions):
    """
    reindex users, items, interactions in case there are some values missing or duplicates in between
    :param users: pd.DataFrame
    :param items: pd.DataFrame
    :param interactions: pd.DataFrame
    :return: same
    """
    print('Reindexing dataframes...')
    unique_uids = users.uid.unique()
    unique_iids = items.iid.unique()

    num_users = unique_uids.shape[0]
    num_movies = unique_iids.shape[0]

    raw_uids = np.array(unique_uids, dtype=np.int)
    raw_iids = np.array(unique_iids, dtype=np.int)
    uids = np.arange(num_users)
    iids = np.arange(num_movies)

    users['uid'] = uids
    items['iid'] = iids

    raw_uid2uid = {raw_uid: uid for raw_uid, uid in zip(raw_uids, uids)}
    raw_iid2iid = {raw_iid: iid for raw_iid, iid in zip(raw_iids, iids)}

    rating_uids = np.array(interactions.uid, dtype=np.int)
    rating_iids = np.array(interactions.iid, dtype=np.int)
    rating_uids = [raw_uid2uid[rating_uid] for rating_uid in rating_uids]
    rating_iids = [raw_iid2iid[rating_iid] for rating_iid in rating_iids]
    interactions['uid'] = rating_uids
    interactions['iid'] = rating_iids
    print('Reindex done!')

    return users, items, interactions


def drop_infrequent_concept_from_str(df, concept_name, num_occs):
    concept_strs = [concept_str for concept_str in df[concept_name]]
    duplicated_concept = [concept_str.split(',') for concept_str in concept_strs]
    duplicated_concept = list(itertools.chain.from_iterable(duplicated_concept))
    writer_counter_dict = Counter(duplicated_concept)
    del writer_counter_dict['']
    del writer_counter_dict['N/A']
    unique_concept = [k for k, v in writer_counter_dict.items() if v >= num_occs]
    concept_strs = [
        ','.join([concept for concept in concept_str.split(',') if concept in unique_concept])
        for concept_str in concept_strs
    ]
    df[concept_name] = concept_strs
    return df


def generate_graph_data(
        users, items, ratings
):
    """
    Entitiy node include (gender, occupation, genres)
    num_nodes = num_users + num_items + num_genders + num_occupation + num_ages + num_genres + num_years + num_directors + num_actors + num_writers
    """
    def get_concept_num_from_str(df, concept_name):
        concept_strs = [concept_str.split(',') for concept_str in df[concept_name]]
        concepts = set(itertools.chain.from_iterable(concept_strs))
        concepts.remove('')
        num_concepts = len(concepts)
        return list(concepts), num_concepts

    #########################  Create dataset property dict  #########################
    dataset_property_dict = {}
    dataset_property_dict['users'] = users
    dataset_property_dict['items'] = items
    dataset_property_dict['ratings'] = ratings

    #########################  Define entities  #########################
    num_users = users.shape[0]
    num_items = items.shape[0]
    dataset_property_dict['num_users'] = num_users
    dataset_property_dict['num_items'] = num_items

    unique_genders = list(users.gender.unique())
    num_genders = len(unique_genders)

    unique_occupations = list(users.occupation.unique())
    num_occupations = len(unique_occupations)

    unique_ages = list(users.age.unique())
    num_ages = len(unique_ages)

    unique_genres = list(items.keys()[3:20])
    num_genres = len(unique_genres)

    unique_years = list(items.year.unique())
    num_years = len(unique_years)

    unique_directors, num_directors = get_concept_num_from_str(items, 'directors')
    unique_actors, num_actors = get_concept_num_from_str(items, 'actors')
    unique_writers, num_writers = get_concept_num_from_str(items, 'writers')

    dataset_property_dict['unique_genders'] = unique_genders
    dataset_property_dict['num_genders'] = num_genders
    dataset_property_dict['unique_occupations'] = unique_occupations
    dataset_property_dict['num_occupations'] = num_occupations
    dataset_property_dict['unique_ages'] = unique_ages
    dataset_property_dict['num_ages'] = num_ages
    dataset_property_dict['unique_genres'] = unique_genres
    dataset_property_dict['num_genres'] = num_genres
    dataset_property_dict['unique_years'] = unique_years
    dataset_property_dict['num_years'] = num_years
    dataset_property_dict['unique_directors'] = unique_directors
    dataset_property_dict['num_directors'] = num_directors
    dataset_property_dict['unique_actors'] = unique_actors
    dataset_property_dict['num_actors'] = num_actors
    dataset_property_dict['unique_writers'] = unique_writers
    dataset_property_dict['num_writers'] = num_writers


    #########################  Define number of entities  #########################
    num_nodes = num_users + num_items + num_genders + num_occupations + num_ages + num_genres + num_years + \
                num_directors + num_actors + num_writers
    num_node_types = 10
    dataset_property_dict['num_nodes'] = num_nodes
    dataset_property_dict['num_node_types'] = num_node_types

    #########################  Define entities to node id map  #########################
    nid2e_dict = {}
    acc = 0
    uid2nid = {uid: i + acc for i, uid in enumerate(users['uid'])}
    for i, uid in enumerate(users['uid']):
        nid2e_dict[i + acc] = ('uid', uid)
    acc += num_users
    iid2nid = {iid: i + acc for i, iid in enumerate(items['iid'])}
    for i, iid in enumerate(items['iid']):
        nid2e_dict[i + acc] = ('iid', iid)
    acc += num_items
    gender2nid = {gender: i + acc for i, gender in enumerate(unique_genders)}
    for i, gender in enumerate(unique_genders):
        nid2e_dict[i + acc] = ('gender', gender)
    acc += num_genders
    occ2nid = {occupation: i + acc for i, occupation in enumerate(unique_occupations)}
    for i, occ in enumerate(unique_occupations):
        nid2e_dict[i + acc] = ('occ', occ)
    acc += num_occupations
    age2nid = {age: i + acc for i, age in enumerate(unique_ages)}
    for i, age in enumerate(unique_ages):
        nid2e_dict[i + acc] = ('age', age)
    acc += num_ages
    genre2nid = {genre: i + acc for i, genre in enumerate(unique_genres)}
    for i, genre in enumerate(unique_genres):
        nid2e_dict[i + acc] = ('genre', genre)
    acc += num_genres
    year2nid = {year: i + acc for i, year in enumerate(unique_years)}
    for i, year in enumerate(unique_years):
        nid2e_dict[i + acc] = ('year', year)
    acc += num_years
    director2nid = {director: i + acc for i, director in enumerate(unique_directors)}
    for i, director in enumerate(unique_directors):
        nid2e_dict[i + acc] = ('director', director)
    acc += num_directors
    actor2nid = {actor: i + acc for i, actor in enumerate(unique_actors)}
    for i, actor in enumerate(unique_actors):
        nid2e_dict[i + acc] = ('actor', actor)
    acc += num_actors
    writer2nid = {writer: i + acc for i, writer in enumerate(unique_writers)}
    for i, writer in enumerate(unique_writers):
        nid2e_dict[i + acc] = ('writer', writer)
    e2nid_dict = {'uid': uid2nid, 'iid': iid2nid, 'gender': gender2nid, 'occ': occ2nid, 'age': age2nid, 'genre': genre2nid,
             'year': year2nid, 'director': director2nid, 'actor': actor2nid, 'writer': writer2nid}
    dataset_property_dict['e2nid_dict'] = e2nid_dict

    #########################  create graphs  #########################
    edge_index_nps = {}
    print('Creating user property edges...')
    u_nids = [e2nid_dict['uid'][uid] for uid in users.uid]
    gender_nids = [e2nid_dict['gender'][gender] for gender in users.gender]
    gender2user_edge_index_np = np.vstack((np.array(gender_nids), np.array(u_nids)))
    occ_nids = [e2nid_dict['occ'][occ] for occ in users.occupation]
    occ2user_edge_index_np = np.vstack((np.array(occ_nids), np.array(u_nids)))
    age_nids = [e2nid_dict['age'][age] for age in users.age]
    age2user_edge_index_np = np.vstack((np.array(age_nids), np.array(u_nids)))
    edge_index_nps['gender2user'] = gender2user_edge_index_np
    edge_index_nps['occ2user'] = occ2user_edge_index_np
    edge_index_nps['age2user'] = age2user_edge_index_np

    print('Creating item property edges...')
    i_nids = [e2nid_dict['iid'][iid] for iid in items.iid]
    year_nids = [e2nid_dict['year'][year] for year in items.year]
    year2item_edge_index_np = np.vstack((np.array(year_nids), np.array(i_nids)))

    genre_nids = []
    i_nids = []
    for genre in unique_genres:
        iids = items.iid[items[genre]]
        i_nids += [e2nid_dict['iid'][iid] for iid in iids]
        genre_nids += [e2nid_dict['genre'][genre] for _ in range(iids.shape[0])]
    genre2item_edge_index_np = np.vstack((np.array(genre_nids), np.array(i_nids)))

    i_nids = [e2nid_dict['iid'][iid] for iid in items.iid]
    directors_list = [
        [director for director in directors.split(',') if director != '']
        for directors in items.directors
    ]
    directors_nids = [[e2nid_dict['director'][director] for director in directors] for directors in directors_list]
    directors_nids = list(itertools.chain.from_iterable(directors_nids))
    d_i_nids = [[i_nid for _ in range(len(directors_list[idx]))] for idx, i_nid in enumerate(i_nids)]
    d_i_nids = list(itertools.chain.from_iterable(d_i_nids))
    director2item_edge_index_np = np.vstack((np.array(directors_nids), np.array(d_i_nids)))

    actors_list = [
        [actor for actor in actors.split(',') if actor != '']
        for actors in items.actors
    ]
    actor_nids = [[e2nid_dict['actor'][actor] for actor in actors] for actors in actors_list]
    actor_nids = list(itertools.chain.from_iterable(actor_nids))
    a_i_nids = [[i_nid for _ in range(len(actors_list[idx]))] for idx, i_nid in enumerate(i_nids)]
    a_i_nids = list(itertools.chain.from_iterable(a_i_nids))
    actor2item_edge_index_np = np.vstack((np.array(actor_nids), np.array(a_i_nids)))

    writers_list = [
        [writer for writer in writers.split(',') if writer != '']
        for writers in items.writers
    ]
    writer_nids = [[e2nid_dict['writer'][writer] for writer in writers] for writers in writers_list]
    writer_nids = list(itertools.chain.from_iterable(writer_nids))
    w_i_nids = [[i_nid for _ in range(len(writers_list[idx]))] for idx, i_nid in enumerate(i_nids)]
    w_i_nids = list(itertools.chain.from_iterable(w_i_nids))
    writer2item_edge_index_np = np.vstack((np.array(writer_nids), np.array(w_i_nids)))
    edge_index_nps['year2item'] = year2item_edge_index_np
    edge_index_nps['genre2item'] = genre2item_edge_index_np
    edge_index_nps['director2item'] = director2item_edge_index_np
    edge_index_nps['actor2item'] = actor2item_edge_index_np
    edge_index_nps['writer2item'] = writer2item_edge_index_np

    print('Creating rating property edges...')
    train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map = {}, {}, {}

    user2item_edge_index_np = np.zeros((2, 0))
    pbar = tqdm.tqdm(users.uid, total=users.uid.shape[0])
    for uid in pbar:
        pbar.set_description('Creating the edges for the user {}'.format(uid))
        uid_ratings = ratings[ratings.uid == uid].sort_values('timestamp')
        uid_iids = uid_ratings[['iid']].to_numpy().reshape(-1)

        unid = e2nid_dict['uid'][uid]
        train_pos_uid_iids = list(uid_iids[:-1])  # Use leave one out setup
        train_pos_uid_inids = [e2nid_dict['iid'][iid] for iid in train_pos_uid_iids]
        test_pos_uid_iids = list(uid_iids[-1:])
        test_pos_uid_inids = [e2nid_dict['iid'][iid] for iid in test_pos_uid_iids]
        neg_uid_iids = list(set(items.iid) - set(uid_iids))
        neg_uid_inids = [e2nid_dict['iid'][iid] for iid in neg_uid_iids]

        train_pos_unid_inid_map[unid] = train_pos_uid_inids
        test_pos_unid_inid_map[unid] = test_pos_uid_inids
        neg_unid_inid_map[unid] = neg_uid_inids

        unid_user2item_edge_index_np = np.array(
            [[unid for _ in range(len(train_pos_uid_inids))], train_pos_uid_inids]
        )
        user2item_edge_index_np = np.hstack([user2item_edge_index_np, unid_user2item_edge_index_np])
    edge_index_nps['user2item'] = user2item_edge_index_np

    dataset_property_dict['edge_index_nps'] = edge_index_nps
    dataset_property_dict['train_pos_unid_inid_map'], dataset_property_dict['test_pos_unid_inid_map'], \
            dataset_property_dict['neg_unid_inid_map'] = \
        train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map

    print('Building the item occurrence map...')
    item_nid_occs = {}
    for iid in items.iid:
        try:
            item_nid_occs[e2nid_dict['iid'][iid]] = ratings[ratings.iid == iid].iloc[0]['movie_count']
        except:
            pass
    dataset_property_dict['item_nid_occs'] = item_nid_occs

    return dataset_property_dict


class MovieLens(Dataset):
    url = 'http://files.grouplens.org/datasets/movielens/'

    def __init__(self,
                 root,
                 name,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 **kwargs):

        self.name = name.lower()
        assert self.name in ['1m']
        self.num_core = kwargs['num_core']
        self.num_feat_core = kwargs['num_feat_core']
        self.seed = kwargs['seed']
        self.num_negative_samples = kwargs['num_negative_samples']
        self.suffix = self.build_suffix()
        self.loss_type = kwargs['loss_type']
        self._negative_sampling = kwargs['_negative_sampling']

        super(MovieLens, self).__init__(root, transform, pre_transform, pre_filter)

        with open(self.processed_paths[0], 'rb') as f:  # Read the class property
            dataset_property_dict = pickle.load(f)
        for k, v in dataset_property_dict.items():
            self[k] = v
        self.train_edge_index = torch.from_numpy(self.edge_index_nps['user2item'].T).long()
        self.num_pos_train_edges = self.train_edge_index.shape[0]

        if self.loss_type == 'BCE':
            self.length = self.num_pos_train_edges * (self.num_negative_samples + 1)
        elif self.loss_type == 'BPR':
            self.length = self.num_pos_train_edges * self.num_negative_samples

        print('Dataset loaded!')

    @property
    def raw_file_names(self):
        return 'ml-{}.zip'.format(self.name.lower())

    @property
    def processed_file_names(self):
        return ['dataset{}.pkl'.format(self.suffix)]

    def download(self):
        path = download_url(self.url + self.raw_file_names, self.raw_dir)
        extract_zip(path, self.raw_dir)

    def process(self):
        unzip_raw_dir = join(self.raw_dir, 'ml-{}'.format(self.name))

        # parser files
        if isfile(join(self.processed_dir, 'movies.csv')) and isfile(join(self.processed_dir, 'ratings.csv')) and isfile(join(self.processed_dir, 'users.csv')):
            print('Read data frame!')
            users = pd.read_csv(join(self.processed_dir, 'users.csv'), sep=';')
            items = pd.read_csv(join(self.processed_dir, 'movies.csv'), sep=';')
            ratings = pd.read_csv(join(self.processed_dir, 'ratings.csv'), sep=';')
            users = users.fillna('')
            items = items.fillna('')
            ratings = ratings.fillna('')
        else:
            print('Data frame not found in {}! Read from raw data!'.format(self.processed_dir))
            users, items, ratings = parse_ml(unzip_raw_dir)

            print('Preprocessing...')
            # remove duplications
            users = users.drop_duplicates()
            items = items.drop_duplicates()
            ratings = ratings.drop_duplicates()
            if users.shape[0] != users.uid.unique().shape[0] or items.shape[0] != items.iid.unique().shape[0]:
                raise ValueError('Duplicates in dfs.')

            # Compute the movie and user counts
            item_count = ratings['iid'].value_counts()
            user_count = ratings['uid'].value_counts()
            item_count.name = 'movie_count'
            user_count.name = 'user_count'
            ratings = ratings.join(item_count, on='iid')
            ratings = ratings.join(user_count, on='uid')

            # Remove infrequent users and item in ratings
            ratings = ratings[ratings.movie_count > self.num_core]
            ratings = ratings[ratings.user_count > self.num_core]

            # Sync the user and item dataframe
            users = users[users.uid.isin(ratings['uid'].unique())]
            items = items[items.iid.isin(ratings['iid'].unique())]
            ratings = ratings[ratings.iid.isin(items['iid'].unique())]
            ratings = ratings[ratings.uid.isin(users['uid'].unique())]

            # Reindex the uid and iid in case of missing values
            users, items, ratings = reindex_df(users, items, ratings)

            # Discretized year
            years = items.year.to_numpy()
            min_year = min(years)
            max_year = max(years)
            num_years = (max_year - min_year) // 10
            discretized_years = [min_year + i * 10 for i in range(num_years + 1)]
            for i, discretized_year in enumerate(discretized_years):
                if i != len(discretized_years) - 1:
                    years[(discretized_year <= years) & (years < discretized_years[i + 1])] = str(discretized_year)
                else:
                    years[discretized_year <= years] = str(discretized_year)
            items['year'] = years

            # Drop the infrequent writer, actor and directors
            items = drop_infrequent_concept_from_str(items, 'writers', self.num_feat_core)
            items = drop_infrequent_concept_from_str(items, 'directors', self.num_feat_core)
            items = drop_infrequent_concept_from_str(items, 'actors', self.num_feat_core)

            print('Preprocessing done.')

            save_df(users, join(self.processed_dir, 'users.csv'))
            save_df(items, join(self.processed_dir, 'movies.csv'))
            save_df(ratings, join(self.processed_dir, 'ratings.csv'))

        dataset_property_dict = generate_graph_data(users, items, ratings)

        with open(self.processed_paths[0], 'wb') as f:
            pickle.dump(dataset_property_dict, f)

    def build_suffix(self):
        suffixes = []
        suffixes.append('core_{}'.format(self.num_core))
        suffixes.append('featcore_{}'.format(self.num_feat_core))
        suffixes.append('seed_{}'.format(self.seed))
        if not suffixes:
            suffix = ''
        else:
            suffix = '_'.join(suffixes)
        return '_' + suffix

    def negative_sampling(self):
        if self.loss_type == 'BCE':
            pos_train_edge = torch.cat(
                [
                    self.train_edge_index,
                    torch.ones((self.num_pos_train_edges, 1)).long()
                ],
                dim=-1
            )

            u_nids = self.train_edge_index[:, 0].tolist()
            negative_inids = []
            p_bar = tqdm.tqdm(u_nids)
            for u_nid in p_bar:
                negative_inids.append(
                    self._negative_sampling(
                        u_nid,
                        self.num_negative_samples,
                        (
                            self.train_pos_unid_inid_map,
                            self.test_pos_unid_inid_map,
                            self.neg_unid_inid_map
                        ),
                        self.item_nid_occs
                    )
                )
                p_bar.set_description('Negative sampling...')
            negative_inids_t = torch.from_numpy(np.vstack(negative_inids).reshape(-1, 1))
            negative_train_edges = torch.cat(
                [
                    self.train_edge_index[:, 0].repeat(self.num_negative_samples).reshape(-1, 1),
                    negative_inids_t,
                    torch.zeros((self.num_pos_train_edges * self.num_negative_samples, 1)).long()
                ],
                dim=-1
            )

            train_data = torch.cat([pos_train_edge, negative_train_edges], dim=0)
        elif self.loss_type == 'BPR':
            u_nids = self.train_edge_index[:, 0].tolist()
            negative_inids = []
            p_bar = tqdm.tqdm(u_nids)
            for u_nid in p_bar:
                negative_inids.append(
                    self._negative_sampling(
                        u_nid,
                        self.num_negative_samples,
                        (
                            self.train_pos_unid_inid_map,
                            self.test_pos_unid_inid_map,
                            self.neg_unid_inid_map
                        ),
                        self.item_nid_occs
                    )
                )
                p_bar.set_description('Negative sampling...')
            negative_inids_t = torch.from_numpy(np.vstack(negative_inids).reshape(-1, 1))

            train_edge_index_t = self.train_edge_index.repeat(1, self.num_negative_samples).view(-1, 2)
            train_data = torch.cat([train_edge_index_t, negative_inids_t], dim=-1)
        else:
            raise NotImplementedError('No negateive sampling for model type: {}.'.format(self.loss_type))
        shuffle_idx = torch.randperm(train_data.shape[0])
        self.train_data = train_data[shuffle_idx]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        r"""Gets the data object at index :obj:`idx` and transforms it (in case
        a :obj:`self.transform` is given).
        In case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a list, a
        tuple, a  LongTensor or a BoolTensor, will return a subset of the
        dataset at the specified indices."""
        if isinstance(idx, str):
            return getattr(self, idx, None)
        else:
            idx = idx.to_list() if torch.is_tensor(idx) else idx
            return self.train_data[idx]


    def __setitem__(self, key, value):
        """Sets the attribute :obj:`key` to :obj:`value`."""
        if isinstance(key, str):
            setattr(self, key, value)
        else:
            raise NotImplementedError('Assignment can\'t be done outside of constructor')

    def __repr__(self):
        return '{}-{}'.format(self.__class__.__name__, self.name.capitalize())


if __name__ == '__main__':
    import os.path as osp

    root = osp.join('.', 'tmp', 'ml')
    name = '1m'
    seed = 2020
    dataset = MovieLens(root=root, name='1m', seed=seed)
    dataloader = DataLoader(dataset)
    for u_nids, pos_inids, neg_inids in dataloader:
        pass
    print('stop')

