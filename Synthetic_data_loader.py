import numpy as np
import pandas as pd
import datetime
import torch
import torch.utils
from datetime import timezone


class EventsDataset(torch.utils.data.Dataset):
    '''
    Base class for event datasets
    '''
    def __init__(self, TZ=None):
        self.TZ = TZ  # timezone.utc

    def get_Adjacency(self, multirelations=False):
        return None, None, None

    def __len__(self):
        return self.n_events

    def __getitem__(self, index):

        tpl = self.all_events[index]
        u, v, rel, time_cur = tpl

        # Compute time delta in seconds (t_p - \bar{t}_p_j) that will be fed to W_t
        time_delta_uv = np.zeros((2, 4))  # two nodes x 4 values

        # most recent previous time for all nodes
        time_bar = self.time_bar.copy()
        assert u != v, (tpl, rel)

        u = int(u)
        v = int(v)
        
        for c, j in enumerate([u, v]):
            t = datetime.datetime.fromtimestamp(self.time_bar[j][0], tz=self.TZ)
            if t.toordinal() >= self.FIRST_DATE.toordinal():  # assume no events before FIRST_DATE
                td = time_cur - t
                time_delta_uv[c] = np.array([td.days,  # total number of days, still can be a big number
                                             td.seconds // 3600,  # hours, max 24
                                             (td.seconds // 60) % 60,  # minutes, max 60
                                             td.seconds % 60],  # seconds, max 60
                                            np.float64)
                # assert time_delta_uv.min() >= 0, (index, tpl, time_delta_uv[c], node_global_time[j])
            else:
                raise ValueError('unexpected result', t, self.FIRST_DATE)
            self.time_bar[j] = time_cur.timestamp()  # last time stamp for nodes u and v

        k = self.event_types_num[rel]

        # sanity checks
        assert np.float64(time_cur.timestamp()) == time_cur.timestamp(), (
        np.float64(time_cur.timestamp()), time_cur.timestamp())
        time_cur = np.float64(time_cur.timestamp())
        time_bar = time_bar.astype(np.float64)
        time_cur = torch.from_numpy(np.array([time_cur])).double()
        assert time_bar.max() <= time_cur, (time_bar.max(), time_cur)
        return u, v, time_delta_uv, k, time_bar, time_cur
    

class SyntheticDataset(EventsDataset):

    def __init__(self, split, data_dir=None, link_feat=False):
        super(SyntheticDataset, self).__init__()

        self.rnd = np.random.RandomState(1111)

        graph_df = pd.read_csv(data_dir)
        graph_df = graph_df.sort_values('time')
        test_time = np.quantile(graph_df.time, 0.70)
        sources = graph_df.u.values
        destinations = graph_df.v.values
        event_type = graph_df.k.values

        timestamps = graph_df.time.values
        timestamps_date = np.array(list(map(lambda x: datetime.datetime.fromtimestamp(int(x), tz=None), timestamps)))

        train_mask = timestamps<=test_time
        test_mask = timestamps>test_time

        all_events = list(zip(sources, destinations, event_type, timestamps_date))

        if split == 'train':
            self.all_events = np.array(all_events)[train_mask].tolist()
        elif split == 'test':
            self.all_events = np.array(all_events)[test_mask].tolist()
        else:
            raise ValueError('invalid split', split)

        self.FIRST_DATE = datetime.datetime.fromtimestamp(0)
        self.END_DATE = timestamps_date[-1]
        self.TEST_TIMESLOTS = [datetime.datetime(1970, 1, 1, tzinfo=self.TZ)]

        self.N_nodes = max(sources.max(),destinations.max())+1

        self.n_events = len(self.all_events)

        self.event_types_num = {0: 0, 1:1}

        self.assoc_types = [0]
        self.time_bar = np.full(self.N_nodes, self.FIRST_DATE.timestamp())

        self.A_initial = np.zeros((self.N_nodes, self.N_nodes))

        print('\nA_initial', np.sum(self.A_initial))


    def get_Adjacency(self, multirelations=False):
        if multirelations:
            print('warning: only one relation type, so multirelations are ignored')
        return self.A_initial