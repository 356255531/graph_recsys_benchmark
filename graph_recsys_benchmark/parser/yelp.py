import re
import pandas as pd
import numpy as np
import os
import json
import itertools
from collections import Counter
from torch_geometric.data import extract_tar


def parse_yelp(dir):
    """
    Read the yelp dataset from .tar file
    :param dir: the path to raw tar file (yelp_dataset.tar)
    :return: yelp_business, yelp_review, yelp_user, yelp_checkin, yelp_tip, pandas.DataFrame
    """

    # Importing Yelp business data
    with open(os.path.join(dir, 'yelp_academic_dataset_business.json'), encoding='utf-8') as json_file:
        data = [json.loads(line) for line in json_file]
        yelp_business_df = pd.DataFrame(data)

    # Importing Yelp review data
    with open(os.path.join(dir, 'yelp_academic_dataset_review.json'), encoding='utf-8') as json_file:
        data = [json.loads(line) for line in json_file]
        yelp_review_df = pd.DataFrame(data)

    # Importing Yelp user data
    with open(os.path.join(dir, 'yelp_academic_dataset_user.json'), encoding='utf-8') as json_file:
        data = [json.loads(line) for line in json_file]
        yelp_user_df = pd.DataFrame(data)

    # Importing Yelp checkin data
    with open(os.path.join(dir, 'yelp_academic_dataset_checkin.json'), encoding='utf-8') as json_file:
        data = [json.loads(line) for line in json_file]
        yelp_checkin_df = pd.DataFrame(data)

    # Importing Yelp tip data
    with open(os.path.join(dir, 'yelp_academic_dataset_tip.json'), encoding='utf-8') as json_file:
        data = [json.loads(line) for line in json_file]
        yelp_tip_df = pd.DataFrame(data)

    return yelp_business_df, yelp_user_df, yelp_review_df, yelp_tip_df, yelp_checkin_df