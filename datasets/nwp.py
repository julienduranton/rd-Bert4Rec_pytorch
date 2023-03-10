# NWP: Next Word Prediction

import os
import pickle
import random
from typing import Optional

from torch import LongTensor
from torch.utils.data import Dataset

__all__ = (
    'NWPTrainDataset',
    'NWPEvalDataset',
    'NWPPredictDataset'
)


class NWPTrainDataset(Dataset):

    data_root = 'data'

    def __init__(self,
                 name: str,
                 sequence_len: int = 200,
                 max_num_segments: int = 10,
                 random_cut_prob: float = 0.0,
                 replace_prob: float = 0.02,
                 use_session_token: bool = False,
                 random_seed: Optional[int] = None
                 ):

        # params
        self.name = name
        self.sequence_len = sequence_len
        self.max_num_segments = max_num_segments
        self.random_cut_prob = random_cut_prob
        self.replace_prob = replace_prob
        self.use_session_token = use_session_token
        self.random_seed = random_seed

        # rng
        if random_seed is None:
            self.rng = random.Random()
        else:
            self.rng = random.Random(random_seed)

        # load data
        with open(os.path.join(self.data_root, name, 'iid2iindex.pkl'), 'rb') as fp:
            self.iid2iindex = pickle.load(fp)
        with open(os.path.join(self.data_root, name, 'train.pkl'), 'rb') as fp:
            self.uindex2rows_train = pickle.load(fp)

        # settle down
        self.uindices = list(self.uindex2rows_train.keys())
        self.num_items = len(self.iid2iindex)

        # tokens
        self.padding_token = 0
        self.mask_token = self.num_items + 1
        self.session_token = self.num_items + 2
        self.ignore_label = 0
        self.padding_segment = 0

    def __len__(self):
        return len(self.uindices)

    def __getitem__(self, index):

        # data point
        uindex = self.uindices[index]
        rows = self.uindex2rows_train[uindex]

        # random cut
        if self.rng.random() < self.random_cut_prob:
            rows = rows[:self.rng.randint(2, len(rows))]

        # tokens and labels and stamps
        tokens = []
        labels = []
        segments = []
        stamps = []
        current_segment = 0
        previous_sid = None
        start_stamp = 0
        for i in range(len(rows) - 1):
            current_iindex, current_sid, stamp = rows[i]
            next_iindex, _, _ = rows[i + 1]

            # save it for stamp padding
            if not i:
                start_stamp = stamp

            # when new session
            if current_sid != previous_sid:
                current_segment += 1
                previous_sid = current_sid
                if self.use_session_token:
                    if len(labels):
                        labels[-1] = self.ignore_label
                    tokens.append(self.session_token)
                    labels.append(current_iindex)
                    segments.append(current_segment)
                    stamps.append(stamp)

            # data augmentation: replace
            prob = self.rng.random()
            if prob < self.replace_prob:
                current_iindex = self.rng.randint(1, self.num_items)

            # add item
            tokens.append(current_iindex)
            labels.append(next_iindex)
            segments.append(current_segment)
            stamps.append(stamp)

        # cut
        tokens = tokens[-self.sequence_len:]
        labels = labels[-self.sequence_len:]
        segments = segments[-self.sequence_len:]
        stamps = stamps[-self.sequence_len:]

        # rename segments
        num_sessions = max(segments)
        segments = [max(1, self.max_num_segments - num_sessions + segment) for segment in segments]

        # add paddings
        padding_len = self.sequence_len - len(tokens)
        tokens = [self.padding_token] * padding_len + tokens
        labels = [self.ignore_label] * padding_len + labels
        segments = [self.padding_segment] * padding_len + segments
        stamps = [start_stamp] * padding_len + stamps

        return {
            'tokens': LongTensor(tokens),
            'labels': LongTensor(labels),
            'segments': LongTensor(segments),
            'stamps': LongTensor(stamps),
        }


class NWPEvalDataset(Dataset):

    data_root = 'data'

    def __init__(self,
                 name: str,
                 target: str,  # 'valid', 'test'
                 ns: str,  # 'random', 'popular', 'all'
                 sequence_len: int = 200,
                 max_num_segments: int = 10,
                 use_session_token: bool = False
                 ):

        # params
        self.name = name
        self.target = target
        self.ns = ns
        self.sequence_len = sequence_len
        self.max_num_segments = max_num_segments
        self.use_session_token = use_session_token

        # load data
        with open(os.path.join(self.data_root, name, 'iid2iindex.pkl'), 'rb') as fp:
            self.iid2iindex = pickle.load(fp)
        with open(os.path.join(self.data_root, name, 'train.pkl'), 'rb') as fp:
            self.uindex2rows_train = pickle.load(fp)
        with open(os.path.join(self.data_root, name, 'valid.pkl'), 'rb') as fp:
            self.uindex2rows_valid = pickle.load(fp)
        with open(os.path.join(self.data_root, name, 'test.pkl'), 'rb') as fp:
            self.uindex2rows_test = pickle.load(fp)

        # in case of ns
        if ns != 'all':
            with open(os.path.join(self.data_root, name, f'ns_{ns}.pkl'), 'rb') as fp:
                self.uindex2negatives = pickle.load(fp)

        # settle down
        if target == 'valid':
            self.uindices = list(self.uindex2rows_valid.keys())
        elif target == 'test':
            self.uindices = list(self.uindex2rows_test.keys())
        self.num_items = len(self.iid2iindex)

        # tokens
        self.padding_token = 0
        self.mask_token = self.num_items + 1
        self.session_token = self.num_items + 2
        self.padding_segment = 0

    def __len__(self):
        return len(self.uindices)

    def __getitem__(self, index):

        # data point
        uindex = self.uindices[index]
        rows_train = self.uindex2rows_train[uindex]
        rows_valid = self.uindex2rows_valid[uindex]
        rows_test = self.uindex2rows_test[uindex]

        # prepare rows
        if self.target == 'valid':
            rows_known = rows_train
            rows_eval = rows_valid
        elif self.target == 'test':
            rows_known = rows_train + rows_valid
            rows_eval = rows_test
            

        # tokens and segments
        tokens = []
        segments = []
        stamps = []
        current_segment = 0
        previous_sid = None
        start_stamp = 0
        for i, (iindex, current_sid, stamp) in enumerate(rows_known):

            # save it for stamp padding
            if not i:
                start_stamp = stamp

            # when new session
            if current_sid != previous_sid:
                current_segment += 1
                previous_sid = current_sid
                if self.use_session_token:
                    tokens.append(self.session_token)
                    segments.append(current_segment)
                    stamps.append(stamp)

            # add item
            tokens.append(iindex)
            segments.append(current_segment)
            stamps.append(stamp)

        # get eval row
        answer, next_sid, next_stamp = rows_eval[0]

        # if the next eval is new session
        if next_sid != previous_sid:
            current_segment += 1
            if self.use_session_token:
                tokens.append(self.session_token)
                segments.append(current_segment)
                stamps.append(next_stamp)

        # cut
        tokens = tokens[-self.sequence_len:]
        segments = segments[-self.sequence_len:]
        stamps = stamps[-self.sequence_len:]

        # rename segments
        num_sessions = max(segments)
        segments = [max(1, self.max_num_segments - num_sessions + segment) for segment in segments]

        # add paddings
        padding_len = self.sequence_len - len(tokens)
        tokens = [self.padding_token] * padding_len + tokens
        segments = [self.padding_segment] * padding_len + segments
        stamps = [start_stamp] * padding_len + stamps

        # candidates and labels
        if self.ns != 'all':
            negatives = self.uindex2negatives[uindex]
            cands = [answer] + negatives
            labels = [1] + [0] * len(negatives)
        else:
            cands = list(range(1, self.num_items))
            labels = [0] * self.num_items
            labels[answer - 1] = 1

        return {
            'tokens': LongTensor(tokens),
            'segments': LongTensor(segments),
            'stamps': LongTensor(stamps),
            'cands': LongTensor(cands),
            'labels': LongTensor(labels),
        }

class NWPPredictDataset(Dataset):

    data_root = 'data'

    def __init__(self,
                 df_predict,
                 ui2uindex,
                 name: str,
                 target: str,  # 'valid', 'test'
                 ns: str,  # 'random', 'popular', 'all'
                 sequence_len: int = 200,
                 max_num_segments: int = 10,
                 use_session_token: bool = False
                 ):

        # params
        self.name = name
        self.target = target
        self.ns = ns
        self.sequence_len = sequence_len
        self.max_num_segments = max_num_segments
        self.use_session_token = use_session_token
        self.df = df_predict
        self.df = self.df.drop(columns=['uindex'])
        self.df = self.df.values.tolist()


        # load data
        with open(os.path.join(self.data_root, name, 'iid2iindex.pkl'), 'rb') as fp:
            self.iid2iindex = pickle.load(fp)
        # with open(os.path.join(self.data_root, name, 'train.pkl'), 'rb') as fp:
        #     self.uindex2rows_train = pickle.load(fp)
        # with open(os.path.join(self.data_root, name, 'valid.pkl'), 'rb') as fp:
        #     self.uindex2rows_valid = pickle.load(fp)
        # with open(os.path.join(self.data_root, name, 'test.pkl'), 'rb') as fp:
        #     self.uindex2rows_test = pickle.load(fp)
        
        self.num_items = len(self.iid2iindex)

        # tokens
        self.padding_token = 0
        self.mask_token = self.num_items + 1
        self.session_token = self.num_items + 2
        self.padding_segment = 0

    def __len__(self):
        return 1

    def __getitem__(self, index):

        # data point

        # prepare rows

        rows_known =self.df
            

        # tokens and segments
        tokens = []
        segments = []
        stamps = []
        current_segment = 0
        previous_sid = None
        start_stamp = 0
        for i, (current_sid, stamp, iindex) in enumerate(rows_known):

            # save it for stamp padding
            if not i:
                start_stamp = stamp

            # when new session
            if current_sid != previous_sid:
                current_segment += 1
                previous_sid = current_sid
                if self.use_session_token:
                    tokens.append(self.session_token)
                    segments.append(current_segment)
                    stamps.append(stamp)

            # add item
            tokens.append(iindex)
            segments.append(current_segment)
            stamps.append(stamp)

        # # get eval row
        # answer, next_sid, next_stamp = rows_eval[0]

        # # if the next eval is new session
        # if next_sid != previous_sid:
        #     current_segment += 1
        #     if self.use_session_token:
        #         tokens.append(self.session_token)
        #         segments.append(current_segment)
        #         stamps.append(next_stamp)

        # cut
        tokens = tokens[-self.sequence_len:]
        segments = segments[-self.sequence_len:]
        stamps = stamps[-self.sequence_len:]

        # rename segments
        num_sessions = max(segments)
        segments = [max(1, self.max_num_segments - num_sessions + segment) for segment in segments]

        # add paddings
        padding_len = self.sequence_len - len(tokens)
        tokens = [self.padding_token] * padding_len + tokens
        segments = [self.padding_segment] * padding_len + segments
        stamps = [start_stamp] * padding_len + stamps

        # candidates and labels
        # if self.ns != 'all':
        #     negatives = self.uindex2negatives[uindex]
        #     cands = [answer] + negatives
        #     labels = [1] + [0] * len(negatives)
        # else:
        cands = list(range(1, self.num_items))
        labels = [0] * self.num_items
        # labels[answer - 1] = 1

        return {
            'tokens': LongTensor(tokens),
            'segments': LongTensor(segments),
            'stamps': LongTensor(stamps),
            'cands': LongTensor(cands),
            'labels': LongTensor(labels),
        }
