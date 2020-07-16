import torch
from dateutil.parser import parser
from torch.utils.data import DataLoader
from os.path import join, isfile, isdir
from pathlib import Path
import numpy as np
import pandas as pd
import itertools
from collections import Counter
from iteration_utilities import unique_everseen
import tqdm
import pickle
import re
from shutil import copy

from .dataset import Dataset
from torch_geometric.data import extract_tar
from ..parser import yelp


def reindex_df(business, user, review, tip):
    """
    reindex business, user, review, tip in case there are some values missing or duplicates in between
    :param business: pd.DataFrame
    :param user: pd.DataFrame
    :param review: pd.DataFrame
    :param tip: pd.DataFrame
    :return: same
    """
    print('Reindexing dataframes...')
    unique_bids = business.business_id.unique()
    unique_uids = user.user_id.unique()

    num_bus = unique_bids.shape[0]
    num_users = unique_uids.shape[0]

    raw_bids = np.array(unique_bids, dtype=object)
    raw_uids = np.array(unique_uids, dtype=object)
    bids = np.arange(num_bus)
    uids = np.arange(num_users)

    business['business_id'] = bids
    user['user_id'] = uids

    raw_bid2bid = {raw_bid: bid for raw_bid, bid in zip(raw_bids, bids)}
    raw_uid2uid = {raw_uid: uid for raw_uid, uid in zip(raw_uids, uids)}

    review_bids = np.array(review.business_id, dtype=object)
    review_uids = np.array(review.user_id, dtype=object)
    review_bids = [raw_bid2bid[review_bid] for review_bid in review_bids]
    review_uids = [raw_uid2uid[review_uid] for review_uid in review_uids]
    review['business_id'] = review_bids
    review['user_id'] = review_uids

    tip_bids = np.array(tip.business_id, dtype=object)
    tip_uids = np.array(tip.user_id, dtype=object)
    tip_bids = [raw_bid2bid[tip_bid] for tip_bid in tip_bids]
    tip_uids = [raw_uid2uid[tip_uid] for tip_uid in tip_uids]
    tip['business_id'] = tip_bids
    tip['user_id'] = tip_uids
    print('Reindex done!')

    return business, user, review, tip


def drop_infrequent_concept_from_str(df, concept_name):
    concept_strs = [concept_str if concept_str != None else '' for concept_str in df[concept_name]]
    duplicated_concept = [concept_str.split(', ') for concept_str in concept_strs]
    duplicated_concept = list(itertools.chain.from_iterable(duplicated_concept))
    writer_counter_dict = Counter(duplicated_concept)
    del writer_counter_dict['']
    del writer_counter_dict['N/A']
    unique_concept = [k for k, v in writer_counter_dict.items() if
                      v >= 0.1 * np.max(list(writer_counter_dict.values()))]
    concept_strs = [
        ','.join([concept for concept in concept_str.split(', ') if concept in unique_concept])
        for concept_str in concept_strs
    ]
    df[concept_name] = concept_strs
    return df


def generate_graph_data(
        business, user, review, tip
):
    """
    Entitiy node include (business, user, review, tip)
    num_nodes = num_users + num_items + num_genders + num_occupation + num_ages + num_genres + num_years + num_directors + num_actors + num_writers
    """
    def get_concept_num_from_str(df, concept_name):
        if (concept_name == 'friends'):
            concept_strs = [concept_str.split(', ') for concept_str in df[concept_name]]
            concepts = set(itertools.chain.from_iterable(concept_strs))
            unique_uids = list(df.user_id.unique())
            concepts = list(set(concepts).difference(unique_uids))
        else:
            concept_strs = [concept_str.split(',') for concept_str in df[concept_name]]
            concepts = set(itertools.chain.from_iterable(concept_strs))
            concepts.remove('')
        num_concepts = len(concepts)
        return list(concepts), num_concepts

    #########################  Create dataset property dict  #########################
    dataset_property_dict = {}
    dataset_property_dict['business'] = business
    dataset_property_dict['user'] = user
    dataset_property_dict['review'] = review
    dataset_property_dict['tip'] = tip

    #########################  Define entities  #########################
    num_bus = business.shape[0]
    num_users = user.shape[0]
    dataset_property_dict['num_bus'] = num_bus
    dataset_property_dict['num_users'] = num_users

    unique_bus_names = list(business.name.unique())
    num_bus_names = len(unique_bus_names)

    unique_bus_city = list(business.city.unique())
    num_bus_city = len(unique_bus_city)

    unique_bus_state = list(business.state.unique())
    num_bus_state = len(unique_bus_state)

    unique_bus_stars = list(business.stars.unique())
    num_bus_stars = len(unique_bus_stars)

    unique_bus_reviewcount = list(business.review_count.unique())
    num_bus_reviewcount = len(unique_bus_reviewcount)

    unique_bus_isopen = list(business.is_open.unique())
    num_bus_isopen = len(unique_bus_isopen)

    unique_bus_attributes, num_bus_attributes = get_concept_num_from_str(business, 'attributes')
    unique_bus_categories, num_bus_categories = get_concept_num_from_str(business, 'categories')

    unique_bus_time = list(business.keys()[13:19])
    num_bus_time = len(unique_bus_time)

    unique_bus_checkincount = list(business.checkin_count.unique())
    num_bus_checkincount = len(unique_bus_checkincount)

    unique_user_names = list(user.name.unique())
    num_user_names = len(unique_user_names)

    unique_user_reviewcount = list(user.review_count.unique())
    num_user_reviewcount = len(unique_user_reviewcount)

    unique_user_startdate = set(list(stdt[:4] for stdt in user.yelping_since))
    num_user_startdate = len(unique_user_startdate)

    unique_user_friends, num_user_friends = get_concept_num_from_str(user, 'friends')

    unique_user_fans = list(user.fans.unique())
    num_user_fans = len(unique_user_fans)

    unique_user_elite, num_user_elite = get_concept_num_from_str(user, 'elite')

    unique_user_averagestars = list(user.average_stars.unique())
    num_user_averagestars = len(unique_user_averagestars)

    dataset_property_dict['unique_bus_names'] = unique_bus_names
    dataset_property_dict['num_bus_names'] = num_bus_names
    dataset_property_dict['unique_bus_city'] = unique_bus_city
    dataset_property_dict['num_bus_city'] = num_bus_city
    dataset_property_dict['unique_bus_state'] = unique_bus_state
    dataset_property_dict['num_bus_state'] = num_bus_state
    dataset_property_dict['unique_bus_stars'] = unique_bus_stars
    dataset_property_dict['num_bus_stars'] = num_bus_stars
    dataset_property_dict['unique_bus_reviewcount'] = unique_bus_reviewcount
    dataset_property_dict['num_bus_reviewcount'] = num_bus_reviewcount
    dataset_property_dict['unique_bus_isopen'] = unique_bus_isopen
    dataset_property_dict['num_bus_isopen'] = num_bus_isopen
    dataset_property_dict['unique_bus_attributes'] = unique_bus_attributes
    dataset_property_dict['num_bus_attributes'] = num_bus_attributes
    dataset_property_dict['unique_bus_categories'] = unique_bus_categories
    dataset_property_dict['num_bus_categories'] = num_bus_categories
    dataset_property_dict['unique_bus_time'] = unique_bus_time
    dataset_property_dict['num_bus_time'] = num_bus_time
    dataset_property_dict['unique_bus_checkincount'] = unique_bus_checkincount
    dataset_property_dict['num_bus_checkincount'] = num_bus_checkincount
    dataset_property_dict['unique_user_names'] = unique_user_names
    dataset_property_dict['num_user_names'] = num_user_names
    dataset_property_dict['unique_user_reviewcount'] = unique_user_reviewcount
    dataset_property_dict['num_user_reviewcount'] = num_user_reviewcount
    dataset_property_dict['unique_user_startdate'] = unique_user_startdate
    dataset_property_dict['num_user_startdate'] = num_user_startdate
    dataset_property_dict['unique_user_friends'] = unique_user_friends
    dataset_property_dict['num_user_friends'] = num_user_friends
    dataset_property_dict['unique_user_fans'] = unique_user_fans
    dataset_property_dict['num_user_fans'] = num_user_fans
    dataset_property_dict['unique_user_elite'] = unique_user_elite
    dataset_property_dict['num_user_elite'] = num_user_elite
    dataset_property_dict['unique_user_averagestars'] = unique_user_averagestars
    dataset_property_dict['num_user_averagestars'] = num_user_averagestars

    #########################  Define number of entities  #########################
    num_nodes = num_bus + num_users + num_bus_names + num_bus_city + num_bus_state + num_bus_stars + num_bus_reviewcount + \
                num_bus_isopen + num_bus_attributes + num_bus_categories + num_bus_time + num_bus_checkincount + \
                num_user_names + num_user_reviewcount + num_user_startdate + num_user_friends + \
                num_user_fans + num_user_elite + num_user_averagestars
    num_node_types = 19
    dataset_property_dict['num_nodes'] = num_nodes
    dataset_property_dict['num_node_types'] = num_node_types

    #########################  Define entities to node id map  #########################
    nid2e_dict = {}
    acc = 0
    bid2nid = {bid: i + acc for i, bid in enumerate(business['business_id'])}
    for i, bid in enumerate(business['business_id']):
        nid2e_dict[i + acc] = ('bid', bid)
    acc += num_bus
    uid2nid = {uid: i + acc for i, uid in enumerate(user['user_id'])}
    for i, uid in enumerate(user['user_id']):
        nid2e_dict[i + acc] = ('uid', uid)
    acc += num_users
    busname2nid = {busname: i + acc for i, busname in enumerate(unique_bus_names)}
    for i, busname in enumerate(unique_bus_names):
        nid2e_dict[i + acc] = ('busname', busname)
    acc += num_bus_names
    buscity2nid = {buscity: i + acc for i, buscity in enumerate(unique_bus_city)}
    for i, buscity in enumerate(unique_bus_city):
        nid2e_dict[i + acc] = ('buscity', buscity)
    acc += num_bus_city
    busstate2nid = {busstate: i + acc for i, busstate in enumerate(unique_bus_state)}
    for i, busstate in enumerate(unique_bus_state):
        nid2e_dict[i + acc] = ('busstate', busstate)
    acc += num_bus_state
    busstars2nid = {busstars: i + acc for i, busstars in enumerate(unique_bus_stars)}
    for i, busstars in enumerate(unique_bus_stars):
        nid2e_dict[i + acc] = ('busstars', busstars)
    acc += num_bus_stars
    busreviewcount2nid = {busreviewcount: i + acc for i, busreviewcount in enumerate(unique_bus_reviewcount)}
    for i, busreviewcount in enumerate(unique_bus_reviewcount):
        nid2e_dict[i + acc] = ('busreviewcount', busreviewcount)
    acc += num_bus_reviewcount
    busisopen2nid = {busisopen: i + acc for i, busisopen in enumerate(unique_bus_isopen)}
    for i, busisopen in enumerate(unique_bus_isopen):
        nid2e_dict[i + acc] = ('busisopen', busisopen)
    acc += num_bus_isopen
    busattributes2nid = {busattributes: i + acc for i, busattributes in enumerate(unique_bus_attributes)}
    for i, busattributes in enumerate(unique_bus_attributes):
        nid2e_dict[i + acc] = ('busattributes', busattributes)
    acc += num_bus_attributes
    buscategories2nid = {buscategories: i + acc for i, buscategories in enumerate(unique_bus_categories)}
    for i, buscategories in enumerate(unique_bus_categories):
        nid2e_dict[i + acc] = ('buscategories', buscategories)
    acc += num_bus_categories
    bustime2nid = {bustime: i + acc for i, bustime in enumerate(unique_bus_time)}
    for i, bustime in enumerate(unique_bus_time):
        nid2e_dict[i + acc] = ('bustime', bustime)
    acc += num_bus_time
    buscheckincount2nid = {buscheckincount: i + acc for i, buscheckincount in enumerate(unique_bus_checkincount)}
    for i, buscheckincount in enumerate(unique_bus_checkincount):
        nid2e_dict[i + acc] = ('buscheckincount', buscheckincount)
    acc += num_bus_checkincount
    usernames2nid = {usernames: i + acc for i, usernames in enumerate(unique_user_names)}
    for i, usernames in enumerate(unique_user_names):
        nid2e_dict[i + acc] = ('usernames', usernames)
    acc += num_user_names
    userreviewcount2nid = {userreviewcount: i + acc for i, userreviewcount in enumerate(unique_user_reviewcount)}
    for i, userreviewcount in enumerate(unique_user_reviewcount):
        nid2e_dict[i + acc] = ('userreviewcount', userreviewcount)
    acc += num_user_reviewcount
    userstartdate2nid = {userstartdate: i + acc for i, userstartdate in enumerate(unique_user_startdate)}
    for i, userstartdate in enumerate(unique_user_startdate):
        nid2e_dict[i + acc] = ('userstartdate', userstartdate)
    acc += num_user_startdate
    userfriends2nid = {userfriends: i + acc for i, userfriends in enumerate(unique_user_friends)}
    for i, userfriends in enumerate(unique_user_friends):
        nid2e_dict[i + acc] = ('userfriends', userfriends)
    acc += num_user_friends
    userfans2nid = {userfans: i + acc for i, userfans in enumerate(unique_user_fans)}
    for i, userfans in enumerate(unique_user_fans):
        nid2e_dict[i + acc] = ('userfans', userfans)
    acc += num_user_fans
    userelite2nid = {userelite: i + acc for i, userelite in enumerate(unique_user_elite)}
    for i, userelite in enumerate(unique_user_elite):
        nid2e_dict[i + acc] = ('userelite', userelite)
    acc += num_user_elite
    useraveragestars2nid = {useraveragestars: i + acc for i, useraveragestars in enumerate(unique_user_averagestars)}
    for i, useraveragestars in enumerate(unique_user_averagestars):
        nid2e_dict[i + acc] = ('useraveragestars', useraveragestars)

    e2nid_dict = {'bid': bid2nid, 'uid': uid2nid, 'busname': busname2nid, 'buscity': buscity2nid,
                  'busstate': busstate2nid, 'busstars': busstars2nid,
                  'busreviewcount': busreviewcount2nid, 'busisopen': busisopen2nid, 'busattributes': busattributes2nid,
                  'buscategories': buscategories2nid,
                  'bustime': bustime2nid, 'buscheckincount': buscheckincount2nid, 'usernames': usernames2nid,
                  'userreviewcount': userreviewcount2nid,
                  'userstartdate': userstartdate2nid, 'userfriends': userfriends2nid,
                  'userfans': userfans2nid, 'userelite': userelite2nid, 'useraveragestars': useraveragestars2nid}
    dataset_property_dict['e2nid_dict'] = e2nid_dict

    #########################  create graphs  #########################
    edge_index_nps = {}
    print('Creating business property edges...')
    b_nids = [e2nid_dict['bid'][bid] for bid in business.business_id]
    busname_nids = [e2nid_dict['busname'][busname] for busname in business.name]
    name2bus_edge_index_np = np.vstack((np.array(busname_nids), np.array(b_nids)))
    buscity_nids = [e2nid_dict['buscity'][buscity] for buscity in business.city]
    city2bus_edge_index_np = np.vstack((np.array(buscity_nids), np.array(b_nids)))
    busstate_nids = [e2nid_dict['busstate'][busstate] for busstate in business.state]
    state2bus_edge_index_np = np.vstack((np.array(busstate_nids), np.array(b_nids)))
    busstars_nids = [e2nid_dict['busstars'][busstars] for busstars in business.stars]
    stars2bus_edge_index_np = np.vstack((np.array(busstars_nids), np.array(b_nids)))
    busreviewcount_nids = [e2nid_dict['busreviewcount'][busreviewcount] for busreviewcount in business.review_count]
    reviewcount2bus_edge_index_np = np.vstack((np.array(busreviewcount_nids), np.array(b_nids)))
    busisopen_nids = [e2nid_dict['busisopen'][busisopen] for busisopen in business.is_open]
    isopen2bus_edge_index_np = np.vstack((np.array(busisopen_nids), np.array(b_nids)))

    attributes_list = [
        [attribute for attribute in attributes.split(',') if attribute != '']
        for attributes in business.attributes
    ]
    busattributes_nids = [[e2nid_dict['busattributes'][attribute] for attribute in attributes] for attributes in
                          attributes_list]
    busattributes_nids = list(itertools.chain.from_iterable(busattributes_nids))
    a_b_nids = [[b_nid for _ in range(len(attributes_list[idx]))] for idx, b_nid in enumerate(b_nids)]
    a_b_nids = list(itertools.chain.from_iterable(a_b_nids))
    attributes2bus_edge_index_np = np.vstack((np.array(busattributes_nids), np.array(a_b_nids)))

    categories_list = [
        [category for category in categories.split(',') if category != '']
        for categories in business.categories
    ]
    buscategories_nids = [[e2nid_dict['buscategories'][category] for category in categories] for categories in
                          categories_list]
    buscategories_nids = list(itertools.chain.from_iterable(buscategories_nids))
    c_b_nids = [[b_nid for _ in range(len(categories_list[idx]))] for idx, b_nid in enumerate(b_nids)]
    c_b_nids = list(itertools.chain.from_iterable(c_b_nids))
    categories2bus_edge_index_np = np.vstack((np.array(buscategories_nids), np.array(c_b_nids)))

    bustime_nids = []
    time_b_nids = []
    for time in unique_bus_time:
        bids = business.business_id[business[time]]
        time_b_nids += [e2nid_dict['bid'][bid] for bid in bids]
        bustime_nids += [e2nid_dict['bustime'][time] for _ in range(bids.shape[0])]
    time2bus_edge_index_np = np.vstack((np.array(bustime_nids), np.array(time_b_nids)))

    buscheckincount_nids = [e2nid_dict['buscheckincount'][buscheckincount] for buscheckincount in
                            business.checkin_count]
    checkincount2bus_edge_index_np = np.vstack((np.array(buscheckincount_nids), np.array(b_nids)))

    edge_index_nps['name2bus'] = name2bus_edge_index_np
    edge_index_nps['city2bus'] = city2bus_edge_index_np
    edge_index_nps['state2bus'] = state2bus_edge_index_np
    edge_index_nps['stars2bus'] = stars2bus_edge_index_np
    edge_index_nps['reviewcount2bus'] = reviewcount2bus_edge_index_np
    edge_index_nps['isopen2bus'] = isopen2bus_edge_index_np
    edge_index_nps['attributes2bus'] = attributes2bus_edge_index_np
    edge_index_nps['categories2bus'] = categories2bus_edge_index_np
    edge_index_nps['time2bus'] = time2bus_edge_index_np
    edge_index_nps['checkincount2bus'] = checkincount2bus_edge_index_np

    print('Creating user property edges...')
    u_nids = [e2nid_dict['uid'][uid] for uid in user.user_id]
    usernames_nids = [e2nid_dict['usernames'][usernames] for usernames in user.name]
    names2user_edge_index_np = np.vstack((np.array(usernames_nids), np.array(u_nids)))
    userreviewcount_nids = [e2nid_dict['userreviewcount'][userreviewcount] for userreviewcount in user.review_count]
    reviewcount2user_edge_index_np = np.vstack((np.array(userreviewcount_nids), np.array(u_nids)))
    userstartdate_nids = [e2nid_dict['userstartdate'][userstartdate[:4]] for userstartdate in user.yelping_since]
    startdate2user_edge_index_np = np.vstack((np.array(userstartdate_nids), np.array(u_nids)))

    friends_list = [
        [friend for friend in friends.split(', ') if friend != '']
        for friends in user.friends
    ]
    userfriends_nids = [
        [e2nid_dict['uid'][friend] if friend in e2nid_dict['uid'] else e2nid_dict['userfriends'][friend] for friend in
         friends] for friends in friends_list]
    userfriends_nids = list(itertools.chain.from_iterable(userfriends_nids))
    f_u_nids = [[u_nid for _ in range(len(friends_list[idx]))] for idx, u_nid in enumerate(u_nids)]
    f_u_nids = list(itertools.chain.from_iterable(f_u_nids))
    friends2user_edge_index_pair = [[userfriends_nids[j], f_u_nids[j]] for j in range(len(userfriends_nids))]
    friends2user_edge_index_pair_unique = list(unique_everseen(friends2user_edge_index_pair, key=frozenset))
    friends_edge_index_pair = []
    user_edge_index_pair = []
    for i in range(len(friends2user_edge_index_pair_unique)):
        friends_edge_index_pair.append(friends2user_edge_index_pair_unique[i][0])
        user_edge_index_pair.append(friends2user_edge_index_pair_unique[i][1])
    friends2user_edge_index_np = np.vstack((np.array(friends_edge_index_pair), np.array(user_edge_index_pair)))

    userfans_nids = [e2nid_dict['userfans'][userfans] for userfans in user.fans]
    fans2user_edge_index_np = np.vstack((np.array(userfans_nids), np.array(u_nids)))

    elites_list = [
        [elite for elite in elites.split(',') if elite != '']
        for elites in user.elite
    ]
    userelite_nids = [[e2nid_dict['userelite'][elite] for elite in elites] for elites in elites_list]
    userelite_nids = list(itertools.chain.from_iterable(userelite_nids))
    e_u_nids = [[u_nid for _ in range(len(elites_list[idx]))] for idx, u_nid in enumerate(u_nids)]
    e_u_nids = list(itertools.chain.from_iterable(e_u_nids))
    elite2user_edge_index_np = np.vstack((np.array(userelite_nids), np.array(e_u_nids)))

    useraveragestars_nids = [e2nid_dict['useraveragestars'][useraveragestars] for useraveragestars in
                             user.average_stars]
    averagestars2user_edge_index_np = np.vstack((np.array(useraveragestars_nids), np.array(u_nids)))

    edge_index_nps['names2user'] = names2user_edge_index_np
    edge_index_nps['reviewcount2user'] = reviewcount2user_edge_index_np
    edge_index_nps['startdate2user'] = startdate2user_edge_index_np
    edge_index_nps['friends2user'] = friends2user_edge_index_np
    edge_index_nps['fans2user'] = fans2user_edge_index_np
    edge_index_nps['elite2user'] = elite2user_edge_index_np
    edge_index_nps['averagestars2user'] = averagestars2user_edge_index_np

    print('Creating review and tip property edges...')
    train_pos_bnid_unid_map, test_pos_bnid_unid_map, neg_bnid_unid_map = {}, {}, {}

    bus2user_edge_index_np = np.zeros((2, 0))
    pbar = tqdm.tqdm(business.business_id, total=business.business_id.shape[0])
    for bid in pbar:
        pbar.set_description('Creating the edges for the business {}'.format(bid))
        bid_review = review[review.business_id == bid].sort_values('stars')
        bid_review_uids = bid_review[['user_id']].to_numpy().reshape(-1)
        bid_tip = tip[tip.business_id == bid].sort_values('compliment_count')
        bid_tip_uids = bid_tip[['user_id']].to_numpy().reshape(-1)
        bid_uids = np.unique(np.concatenate((bid_review_uids, bid_tip_uids)))

        bnid = e2nid_dict['bid'][bid]
        train_pos_bid_uids = list(bid_uids[:-1])  # Use leave one out setup
        train_pos_bid_unids = [e2nid_dict['uid'][uid] for uid in train_pos_bid_uids]
        test_pos_bid_uids = list(bid_uids[-1:])
        test_pos_bid_unids = [e2nid_dict['uid'][uid] for uid in test_pos_bid_uids]
        neg_bid_uids = list(set(user.user_id) - set(bid_uids))
        neg_bid_unids = [e2nid_dict['uid'][uid] for uid in neg_bid_uids]

        train_pos_bnid_unid_map[bnid] = train_pos_bid_unids
        test_pos_bnid_unid_map[bnid] = test_pos_bid_unids
        neg_bnid_unid_map[bnid] = neg_bid_unids

        bnid_bus2user_edge_index_np = np.array(
            [[bnid for _ in range(len(train_pos_bid_unids))], train_pos_bid_unids]
        )
        bus2user_edge_index_np = np.hstack([bus2user_edge_index_np, bnid_bus2user_edge_index_np])

    edge_index_nps['bus2user'] = bus2user_edge_index_np

    dataset_property_dict['edge_index_nps'] = edge_index_nps
    dataset_property_dict['train_pos_bnid_unid_map'], dataset_property_dict['test_pos_bnid_unid_map'], \
    dataset_property_dict['neg_bnid_unid_map'] = \
        train_pos_bnid_unid_map, test_pos_bnid_unid_map, neg_bnid_unid_map

    print('Building the user occurrence map...')
    user_nid_occs = {}
    for uid in user.user_id:
        try:
            user_nid_occs[e2nid_dict['uid'][uid]] = review[review.user_id == uid].iloc[0]['user_count'] + tip[tip.user_id == uid].iloc[0]['user_count']
        except:
            pass
    dataset_property_dict['user_nid_occs'] = user_nid_occs

    return dataset_property_dict


class Yelp(Dataset):

    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 **kwargs):

        self.num_core = kwargs['num_core']
        self.seed = kwargs['seed']
        self.num_negative_samples = kwargs['num_negative_samples']
        self.suffix = self.build_suffix()
        self.loss_type = kwargs['loss_type']
        self._negative_sampling = kwargs['_negative_sampling']
        self.dataset = kwargs['dataset']

        super(Yelp, self).__init__(root, transform, pre_transform, pre_filter)

        with open(self.processed_paths[0], 'rb') as f:  # Read the class property
            dataset_property_dict = pickle.load(f)
        for k, v in dataset_property_dict.items():
            self[k] = v
        self.train_edge_index = torch.from_numpy(self.edge_index_nps['bus2user'].T).long()
        self.num_pos_train_edges = self.train_edge_index.shape[0]

        if self.loss_type == 'BCE':
            self.length = self.num_pos_train_edges * (self.num_negative_samples + 1)
        elif self.loss_type == 'BPR':
            self.length = self.num_pos_train_edges * self.num_negative_samples

        print('Dataset loaded!')

    @property
    def raw_file_names(self):
        return 'yelp_dataset.tar'

    @property
    def processed_file_names(self):
        return ['dataset{}.pkl'.format(self.suffix)]

    @property
    def untar_file_path(self):
        return join(self.raw_dir, 'yelp_dataset')

    def download(self):
        if isdir(self.untar_file_path):
            return

        tar_file_path = join(Path(__file__).parents[2], 'datasets', self.dataset, self.raw_file_names)
        copy(tar_file_path, self.raw_dir)
        extract_tar(join(self.raw_dir, self.raw_file_names), self.untar_file_path )

    def process(self):
        # parser files
        if isfile(join(self.processed_dir, 'business.pkl')) and isfile(join(self.processed_dir, 'user.pkl')) and isfile(
                join(self.processed_dir, 'review.pkl')) and isfile(join(self.processed_dir, 'tip.pkl')):
            print('Read data frame!')
            business = pd.read_pickle(join(self.processed_dir, 'business.pkl'))
            user = pd.read_pickle(join(self.processed_dir, 'user.pkl'))
            review = pd.read_pickle(join(self.processed_dir, 'review.pkl'))
            tip = pd.read_pickle(join(self.processed_dir, 'tip.pkl'))
            business = business.fillna('')
            user = user.fillna('')
            review = review.fillna('')
            tip = tip.fillna('')
        else:
            print('Data frame not found in {}! Read from raw data!'.format(self.processed_dir))
            business, user, review, tip, checkin = yelp(self.untar_file_path)

            print('Preprocessing...')
            # Extract business hours
            hours = []
            for hr in business['hours']:
                hours.append(hr) if hr != None else hours.append({})

            df_hours = (
                pd.DataFrame(hours)
                    .fillna(False))

            # Replacing all times with True
            df_hours.where(df_hours == False, True, inplace=True)

            # Filter business categories > 1% of max value
            business = drop_infrequent_concept_from_str(business, 'categories')

            # Extract business attributes
            attributes = []
            for attr_list in business['attributes']:
                attr_dict = {}
                if attr_list != None:
                    for a, b in attr_list.items():
                        if (b.lower() == 'true' or ''.join(re.findall(r"'(.*?)'", b)).lower() in (
                        'outdoor', 'yes', 'allages', '21plus', '19plus', '18plus', 'full_bar', 'beer_and_wine',
                        'yes_free', 'yes_corkage', 'free', 'paid', 'quiet', 'average', 'loud', 'very_loud', 'casual',
                        'formal', 'dressy')):
                            attr_dict[a.strip()] = True
                        elif (b.lower() in ('false', 'none') or ''.join(re.findall(r"'(.*?)'", b)).lower() in (
                        'no', 'none')):
                            attr_dict[a.strip()] = False
                        elif (b[0] != '{'):
                            attr_dict[a.strip()] = True
                        else:
                            for c in b.split(","):
                                attr_dict[a.strip()] = False
                                if (c == '{}'):
                                    attr_dict[a.strip()] = False
                                    break
                                elif (c.split(":")[1].strip().lower() == 'true'):
                                    attr_dict[a.strip()] = True
                                    break
                attributes.append([k for k, v in attr_dict.items() if v == True])

            business['attributes'] = [','.join(map(str, l)) for l in attributes]

            # Concating business df
            business_concat = [business.iloc[:, :-1], df_hours]
            business = pd.concat(business_concat, axis=1)

            # Compute friend counts
            user['friends_count'] = [len(f.split(",")) if f != 'None' else 0 for f in user['friends']]

            # Compute checkin counts
            checkin['checkin_count'] = [len(f.split(",")) if f != 'None' else 0 for f in checkin['date']]

            # Extract business checkin times
            checkin_years = []
            checkin_months = []
            checkin_time = []
            for checkin_list in checkin['date']:
                checkin_years_ar = []
                checkin_months_ar = []
                checkin_time_ar = []
                if checkin_list != '':
                    for chk in checkin_list.split(","):
                        checkin_years_ar.append(chk.strip()[:4])
                        checkin_months_ar.append(chk.strip()[:7])

                        if int(chk.strip()[11:13]) in range(0, 4):
                            checkin_time_ar.append('00-03')
                        elif int(chk.strip()[11:13]) in range(3, 7):
                            checkin_time_ar.append('03-06')
                        elif int(chk.strip()[11:13]) in range(6, 10):
                            checkin_time_ar.append('06-09')
                        elif int(chk.strip()[11:13]) in range(9, 13):
                            checkin_time_ar.append('09-12')
                        elif int(chk.strip()[11:13]) in range(12, 16):
                            checkin_time_ar.append('12-15')
                        elif int(chk.strip()[11:13]) in range(15, 19):
                            checkin_time_ar.append('15-18')
                        elif int(chk.strip()[11:13]) in range(18, 22):
                            checkin_time_ar.append('18-21')
                        elif int(chk.strip()[11:13]) in range(21, 24):
                            checkin_time_ar.append('21-24')

                checkin_years.append(Counter(checkin_years_ar))
                checkin_months.append(Counter(checkin_months_ar))
                checkin_time.append(Counter(checkin_time_ar))

            df_checkin = (pd.concat([
                pd.DataFrame(checkin_years)
                    .fillna('0').sort_index(axis=1),
                pd.DataFrame(checkin_months)
                    .fillna('0').sort_index(axis=1),
                pd.DataFrame(checkin_time)
                    .fillna('0').sort_index(axis=1)], axis=1))

            # Concating checkin df
            checkin_concat = [checkin, df_checkin]
            checkin = pd.concat(checkin_concat, axis=1)

            # Merging business and checkin
            business = pd.merge(business, checkin, on='business_id', how='left').fillna(0)

            # remove duplications
            business = business.drop_duplicates()
            user = user.drop_duplicates()
            review = review.drop_duplicates()
            tip = tip.drop_duplicates()
            checkin = checkin.drop_duplicates()
            if business.shape[0] != business.business_id.unique().shape[0] or user.shape[0] != \
                    user.user_id.unique().shape[0] or review.shape[0] != review.review_id.unique().shape[0] \
                    or checkin.shape[0] != checkin.business_id.unique().shape[0]:
                raise ValueError('Duplicates in dfs.')

            # Compute the business and user counts for review
            bus_count = review['business_id'].value_counts()
            user_count = review['user_id'].value_counts()
            bus_count.name = 'bus_count'
            user_count.name = 'user_count'
            review = review.join(bus_count, on='business_id')
            review = review.join(user_count, on='user_id')

            # Compute the business and user counts for tip
            bus_count = tip['business_id'].value_counts()
            user_count = tip['user_id'].value_counts()
            bus_count.name = 'bus_count'
            user_count.name = 'user_count'
            tip = tip.join(bus_count, on='business_id')
            tip = tip.join(user_count, on='user_id')

            # Remove infrequent business and user in review
            review = review[review.bus_count > self.num_core]
            review = review[review.user_count > self.num_core]

            # Remove infrequent business and user in tip
            tip = tip[tip.bus_count > self.num_core]
            tip = tip[tip.user_count > self.num_core]

            # Sync the business and user dataframe
            for i in range(0,4):
                business = business[business.business_id.isin(review['business_id'].unique())]
                user = user[user.user_id.isin(review['user_id'].unique())]
                review = review[review.user_id.isin(user['user_id'].unique())]
                review = review[review.business_id.isin(business['business_id'].unique())]

                business = business[business.business_id.isin(tip['business_id'].unique())]
                user = user[user.user_id.isin(tip['user_id'].unique())]
                tip = tip[tip.user_id.isin(user['user_id'].unique())]
                tip = tip[tip.business_id.isin(business['business_id'].unique())]

            # Reindex the bid and uid in case of missing values
            business, user, review, tip = reindex_df(business, user, review, tip)

            print('Preprocessing done.')

            business.to_pickle(join(self.processed_dir, 'business.pkl'))
            user.to_pickle(join(self.processed_dir, 'user.pkl'))
            review.to_pickle(join(self.processed_dir, 'review.pkl'))
            tip.to_pickle(join(self.processed_dir, 'tip.pkl'))

        dataset_property_dict = generate_graph_data(business, user, review, tip)

        with open(self.processed_paths[0], 'wb') as f:
            pickle.dump(dataset_property_dict, f)

    def build_suffix(self):
        suffixes = []
        suffixes.append('core_{}'.format(self.num_core))
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

            b_nids = self.train_edge_index[:, 0].tolist()
            negative_unids = []
            p_bar = tqdm.tqdm(b_nids)
            for b_nid in p_bar:
                negative_unids.append(
                    self._negative_sampling(
                        b_nid,
                        self.num_negative_samples,
                        (
                            self.train_pos_bnid_unid_map,
                            self.test_pos_bnid_unid_map,
                            self.neg_bnid_unid_map
                        ),
                        self.user_nid_occs
                    )
                )
                p_bar.set_description('Negative sampling...')
            negative_unids_t = torch.from_numpy(np.vstack(negative_unids).reshape(-1, 1))
            negative_train_edges = torch.cat(
                [
                    self.train_edge_index[:, 0].repeat(self.num_negative_samples).reshape(-1, 1),
                    negative_unids_t,
                    torch.zeros((self.num_pos_train_edges * self.num_negative_samples, 1)).long()
                ],
                dim=-1
            )

            train_data = torch.cat([pos_train_edge, negative_train_edges], dim=0)
        elif self.loss_type == 'BPR':
            b_nids = self.train_edge_index[:, 0].tolist()
            negative_unids = []
            p_bar = tqdm.tqdm(b_nids)
            for b_nid in p_bar:
                negative_unids.append(
                    self._negative_sampling(
                        b_nid,
                        self.num_negative_samples,
                        (
                            self.train_pos_bnid_unid_map,
                            self.test_pos_bnid_unid_map,
                            self.neg_bnid_unid_map
                        ),
                        self.user_nid_occs
                    )
                )
                p_bar.set_description('Negative sampling...')
            negative_unids_t = torch.from_numpy(np.vstack(negative_unids).reshape(-1, 1))

            train_edge_index_t = self.train_edge_index.repeat(1, self.num_negative_samples).view(-1, 2)
            train_data = torch.cat([train_edge_index_t, negative_unids_t], dim=-1)
        else:
            raise NotImplementedError('No negative sampling for model type: {}.'.format(self.loss_type))
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

    root = osp.join('.', 'tmp', 'yelp')
    name = 'Yelp'
    seed = 2020
    dataset = Yelp(root=root, seed=seed)
    dataloader = DataLoader(dataset)
    for u_nids, pos_inids, neg_inids in dataloader:
        pass
    print('stop')