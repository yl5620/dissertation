import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DyRep_update(nn.Module):
    def __init__(self,
                 node_embeddings,
                 A_initial=None,
                 N_surv_samples=5,
                 n_hidden=32,
                 N_hops=2,
                 gamma = 0.5,
                 decay_rate = 0.0,
                 threshold=0.5,
                 with_attributes=False,
                 with_decay=False,
                 new_SA= False,
                 sparse=False,
                 node_degree_global=None,
                 rnd=np.random.RandomState(111)):
        super(DyRep_update, self).__init__()
    
        # initialisations
        self.opt = True
        self.exp = True
        self.rnd = rnd
        self.n_hidden = n_hidden
        self.sparse = sparse
        self.with_attributes = with_attributes
        self.with_decay = with_decay
        self.new_SA = new_SA
        self.N_surv_samples = N_surv_samples
        self.node_degree_global = node_degree_global
        self.N_nodes = A_initial.shape[0]
        if A_initial is not None and len(A_initial.shape) == 2:
            A_initial = A_initial[:, :, None]
        self.n_assoc_types = 1
        self.N_hops = N_hops
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.threshold = threshold

        self.initialize(node_embeddings, A_initial)
        self.W_h = nn.Linear(in_features=n_hidden, out_features=n_hidden)
        self.W_struct = nn.Linear(n_hidden * self.n_assoc_types, n_hidden)
        self.W_rec = nn.Linear(n_hidden, n_hidden)
        self.W_t = nn.Linear(4, n_hidden)

        n_types = 2  # associative and communicative
        d1 = self.n_hidden + (0)
        d2 = self.n_hidden + (0)

        d1 += self.n_hidden
        d2 += self.n_hidden
        self.omega = nn.ModuleList([nn.Linear(d1, 1), nn.Linear(d2, 1)])

        self.psi = nn.Parameter(0.5 * torch.ones(n_types)) 

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)

    def generate_S_from_A(self):
        if isinstance(self.A, np.ndarray):
            self.A = torch.tensor(self.A, dtype=torch.float32)  # Convert A to a tensor if it's a numpy array
        S = self.A.new_empty(self.N_nodes, self.N_nodes, self.n_assoc_types).fill_(0)
        for rel in range(self.n_assoc_types):
            D = torch.sum(self.A[:, :, rel], dim=1).float()
            for i, v in enumerate(torch.nonzero(D, as_tuple=False).squeeze()):
                u = torch.nonzero(self.A[v, :, rel].squeeze(), as_tuple=False).squeeze()
                S[v, u, rel] = 1. / D[v]
        self.S = S
        # Check that values in each row of S add up to 1
        for rel in range(self.n_assoc_types):
            S = self.S[:, :, rel]
            assert torch.sum(S[self.A[:, :, rel] == 0]) < 1e-5, torch.sum(S[self.A[:, :, rel] == 0])

    def initialize(self,node_embeddings, A_initial,keepS=False):
        print('initialize model''s node embeddings and adjacency matrices for %d nodes' % self.N_nodes)
        # Initial embeddings
        if node_embeddings is not None:
            z = np.pad(node_embeddings, ((0, 0), (0, self.n_hidden - node_embeddings.shape[1])), 'constant')
            z = torch.from_numpy(z).float()

        if A_initial is None:
            print('initial random prediction of A')
            A = torch.zeros(self.N_nodes, self.N_nodes, self.n_assoc_types + int(self.sparse))

            for i in range(self.N_nodes):
                for j in range(i + 1, self.N_nodes):
                    if self.sparse:
                        if self.n_assoc_types == 1:
                            pvals = [0.95, 0.05]
                        elif self.n_assoc_types == 2:
                            pvals = [0.9, 0.05, 0.05]
                        elif self.n_assoc_types == 3:
                            pvals = [0.91, 0.03, 0.03, 0.03]
                        elif self.n_assoc_types == 4:
                            pvals = [0.9, 0.025, 0.025, 0.025, 0.025]
                        else:
                            raise NotImplementedError(self.n_assoc_types)
                        ind = np.nonzero(np.random.multinomial(1, pvals))[0][0]
                    else:
                        ind = np.random.randint(0, self.n_assoc_types, size=1)
                    A[i, j, ind] = 1
                    A[j, i, ind] = 1
            assert torch.sum(torch.isnan(A)) == 0, (torch.sum(torch.isnan(A)), A)
            if self.sparse:
                A = A[:, :, 1:]

        else:
            print('A_initial', A_initial.shape)
            A = torch.from_numpy(A_initial).float()
            if len(A.shape) == 2:
                A = A.unsqueeze(2)

        # make these variables part of the model
        self.register_buffer('z', z)
        self.register_buffer('A', A)

        self.A = A  
        if not keepS:
            self.generate_S_from_A()

        self.Lambda_dict = torch.zeros(5000)
        self.time_keys = []

        self.t_p = 0  # global counter of iterations
    
    def check_S(self):
        for rel in range(self.n_assoc_types):
            rows = torch.nonzero(torch.sum(self.A[:, :, rel], dim=1).float())
            # check that the sum in all rows equal 1
            assert torch.all(torch.abs(torch.sum(self.S[:, :, rel], dim=1)[rows] - 1) < 1e-1), torch.abs(torch.sum(self.S[:, :, rel], dim=1)[rows] - 1)

    
    def g_fn(self,z_cat, k, edge_type=None, z2=None):
        if z2 is not None:
            z_cat = torch.cat((z_cat, z2), dim=1)
        else:
            raise NotImplementedError('')
        g = z_cat.new(len(z_cat), 1).fill_(0)
        idx = k <= 0
        if torch.sum(idx) > 0:
            if edge_type is not None:
                z_cat1 = torch.cat((z_cat[idx], edge_type[idx, :self.n_assoc_types]), dim=1)
            else:
                z_cat1 = z_cat[idx]
            g[idx] = self.omega[0](z_cat1)
        idx = k > 0
        if torch.sum(idx) > 0:
            if edge_type is not None:
                z_cat1 = torch.cat((z_cat[idx], edge_type[idx, self.n_assoc_types:]), dim=1)
            else:
                z_cat1 = z_cat[idx]
            g[idx] = self.omega[1](z_cat1)

        g = g.flatten()
        return g
    
    def intensity_rate_lambda(self,z_u, z_v, k):
        z_u = z_u.view(-1, self.n_hidden).contiguous()
        z_v = z_v.view(-1, self.n_hidden).contiguous()
        edge_type = None
        g = 0.5 * (self.g_fn(z_u, (k > 0).long(), edge_type=edge_type, z2=z_v) + self.g_fn(z_v, (k > 0).long(),edge_type=edge_type, z2=z_u))  # make it symmetric, because most events are symmetric
        psi = self.psi[(k > 0).long()]
        g_psi = torch.clamp(g / (psi + 1e-7), -75, 75)  # to prevent overflow
        Lambda = psi * (torch.log(1 + torch.exp(-g_psi)) + g_psi)
        return Lambda
    
    def update_node_embed_wa(self,prev_embed, node1, node2, sim, time_delta_uv):
        # z contains all node embeddings of previous time \bar{t}
        # S also corresponds to previous time stamp, because it's not updated yet based on this event

        node_embed = prev_embed

        node_degree = {} 
        z_new = prev_embed.clone()  # to allow in place changes while keeping gradients
        
        #precompute the N-hop neighbors
        A_float = self.A.squeeze(-1).float()
        A_power = torch.eye(A_float.shape[0])
        extended_neighbors = [A_power.clone()]
        for _ in range(self.N_hops):
            A_power = torch.mm(A_power, A_float)
            extended_neighbors.append((A_power>0).clone())
        h_u_struct = prev_embed.new_zeros((2, self.n_hidden, self.n_assoc_types))
        for c, (v, u, delta_t) in enumerate(zip([node1, node2], [node2, node1], time_delta_uv)):  # i is the other node involved in the event
            node_degree[u] = np.zeros(self.n_assoc_types)
            for rel in range(self.n_assoc_types):
                Neighb_u = torch.zeros(self.A.shape[1],dtype=torch.bool)
                for i in range(1,self.N_hops+1):
                    Neighb_u |=  extended_neighbors[i][u,:] >0
                
                N_neighb = torch.sum(Neighb_u).item()  # number of neighbors for node u
                node_degree[u][rel] = N_neighb
                if N_neighb > 0:  
                    h_prev_i = self.W_h(node_embed[Neighb_u]).view(N_neighb, self.n_hidden)
                    # attention over neighbors
                    q_ui = torch.exp((1-self.gamma)* sim + self.gamma* self.S[u, Neighb_u, rel]).view(N_neighb, 1)
                    q_ui = q_ui / (torch.sum(q_ui) + 1e-7)
                    h_u_struct[c, :, rel] = torch.max(torch.sigmoid(q_ui * h_prev_i), dim=0)[0].view(1, self.n_hidden)

        h1 = self.W_struct(h_u_struct.view(2, self.n_hidden * self.n_assoc_types))

        h2 = self.W_rec(node_embed[[node1, node2], :].view(2, -1))

        h3 = self.W_t(time_delta_uv.float()).view(2, self.n_hidden)

        z_new[[node1, node2], :] = torch.sigmoid(h1 + h2 + h3)
        return node_degree, z_new
        
    
    def update_node_embed(self,prev_embed, node1, node2, time_delta_uv):
        # z contains all node embeddings of previous time \bar{t}
        # S also corresponds to previous time stamp, because it's not updated yet based on this event

        node_embed = prev_embed

        node_degree = {} 
        z_new = prev_embed.clone()  # to allow in place changes while keeping gradients
        
        #precompute the N-hop neighbors
        A_float = self.A.squeeze(-1).float()
        A_power = torch.eye(A_float.shape[0])
        extended_neighbors = [A_power.clone()]
        for _ in range(self.N_hops):
            A_power = torch.mm(A_power, A_float)
            extended_neighbors.append((A_power>0).clone())
        h_u_struct = prev_embed.new_zeros((2, self.n_hidden, self.n_assoc_types))
        for c, (v, u, delta_t) in enumerate(zip([node1, node2], [node2, node1], time_delta_uv)):  # i is the other node involved in the event
            node_degree[u] = np.zeros(self.n_assoc_types)
            for rel in range(self.n_assoc_types):
                Neighb_u = torch.zeros(self.A.shape[1],dtype=torch.bool)
                for i in range(1,self.N_hops+1):
                    Neighb_u |=  extended_neighbors[i][u,:] > 0
                
                N_neighb = torch.sum(Neighb_u).item()  # number of neighbors for node u
                node_degree[u][rel] = N_neighb
                if N_neighb > 0:  # node has no neighbors
                    h_prev_i = self.W_h(node_embed[Neighb_u]).view(N_neighb, self.n_hidden)
                    if self.with_decay == True:
                        if time_delta_uv.ndim == 2 and time_delta_uv.size(0) == 2:
                            time_delta_uv_expanded = time_delta_uv[0].unsqueeze(0).repeat(N_neighb, 1)
                        else:
                            time_delta_uv_expanded = time_delta_uv[Neighb_u]
                        decay_factor = torch.exp(-self.decay_rate * time_delta_uv_expanded)
                        S_expanded = self.S[u, Neighb_u, rel].unsqueeze(1)
                        attention_scores = decay_factor * torch.exp(time_delta_uv_expanded + S_expanded)
                        q_ui = attention_scores.mean(dim=1, keepdim=True)
                        q_ui_expanded = q_ui.expand(-1, self.n_hidden)
                        h_u_struct[c, :, rel] = torch.max(torch.sigmoid(q_ui_expanded * h_prev_i), dim=0)[0].view(1, self.n_hidden)
                    else:
                        # attention over neighbors
                        q_ui = torch.exp(self.S[u, Neighb_u, rel]).view(N_neighb, 1)
                        q_ui = q_ui / (torch.sum(q_ui) + 1e-7)
                        h_u_struct[c, :, rel] = torch.max(torch.sigmoid(q_ui * h_prev_i), dim=0)[0].view(1, self.n_hidden)
                        

        h1 = self.W_struct(h_u_struct.view(2, self.n_hidden * self.n_assoc_types))

        h2 = self.W_rec(node_embed[[node1, node2], :].view(2, -1))
        h3 = self.W_t(time_delta_uv.float()).view(2, self.n_hidden)

        z_new[[node1, node2], :] = torch.sigmoid(h1 + h2 + h3)
        return node_degree, z_new
        
     
    def update_S_A(self, u, v, k, node_degree, lambda_uv_t):
        if self.new_SA == True:
            if k <= 0 :  # Association event
                self.A[u, v, np.abs(k)] = self.A[v, u, np.abs(k)] = 1  # 0 for CloseFriends
            indices = torch.arange(self.N_nodes)
            for rel in range(self.n_assoc_types):
                for j, i in zip([u, v], [v, u]):
                    try:
                        degree = node_degree[j]
                    except:
                        print(list(node_degree.keys()))
                        raise
                    y = self.S[j, :, rel]
                    b = 0 if degree[rel] == 0 else 1. / (float(degree[rel]) + 1e-7)
                    if k > 0 and self.A[u, v, rel] > 0:  # Communication event, Association exists
                        y[i] = b + lambda_uv_t
                    elif k <= 0 and self.A[u, v, rel] > 0:  # Association event
                        if self.node_degree_global[rel][j] == 0:
                            b_prime = 0
                        else:
                            b_prime = 1. / (float(self.node_degree_global[rel][j]) + 1e-7)
                        x = b_prime - b
                        y[i] = b + lambda_uv_t
                        w = (y != 0) & (indices != int(i))
                        y[w] = y[w] - x
                    y /= (torch.sum(y) + 1e-7)  # normalize
                    self.S[j, :, rel] = y
                if k > 0 and self.A[u, v, rel] != 0:
                    if y[i] >= self.threshold:  # Ensure correct tensor comparison
                        self.A[u, v, rel] = self.A[v, u, rel] = 1
        else:
            if k <= 0 :  # Association event
                # do not update in case of latent graph
                self.A[u, v, np.abs(k)] = self.A[v, u, np.abs(k)] = 1  # 0 for CloseFriends, k = -1 for the second relation, so it's abs(k) matrix in self.A
            A = self.A
            indices = torch.arange(self.N_nodes)
            for rel in range(self.n_assoc_types):
                if k > 0 and A[u, v, rel] == 0:  # Communication event, no Association exists
                    continue  # do not update S and A
                else:
                    for j, i in zip([u, v], [v, u]):
                        # i is the "other node involved in the event"
                        try:
                            degree = node_degree[j]
                        except:
                            print(list(node_degree.keys()))
                            raise
                        y = self.S[j, :, rel]
                        # assert torch.sum(torch.isnan(y)) == 0, ('b', j, degree[rel], node_degree_global[rel][j.item()], y)
                        b = 0 if degree[rel] == 0 else 1. / (float(degree[rel]) + 1e-7)
                        if k > 0 and A[u, v, rel] > 0:  # Communication event, Association exists
                            y[i] = b + lambda_uv_t
                        elif k <= 0 and A[u, v, rel] > 0:  # Association event
                            if self.node_degree_global[rel][j] == 0:
                                b_prime = 0
                            else:
                                b_prime = 1. / (float(self.node_degree_global[rel][j]) + 1e-7)
                            x = b_prime - b
                            y[i] = b + lambda_uv_t
                            w = (y != 0) & (indices != int(i))
                            y[w] = y[w] - x
                        y /= (torch.sum(y) + 1e-7)  # normalize
                        self.S[j, :, rel] = y

            return 
    
    # conditional density calculation to predict the next event (the probability of the next event for each pair of nodes)
    def cond_density(self,time_bar,u, v):
        N = self.N_nodes
        if not self.time_keys:  # Checks if time_keys is empty
            print("Warning: time_keys is empty. No operations performed.")
            return torch.zeros((2, self.N_nodes)) 
        s = self.Lambda_dict.new_zeros((2, N))
        #normalize lambda values by dividing by the number of events
        Lambda_sum = torch.cumsum(self.Lambda_dict.flip(0), 0).flip(0)  / len(self.Lambda_dict)
        time_keys_min = self.time_keys[0]
        time_keys_max = self.time_keys[-1]

        indices = []
        l_indices = []
        t_bar_min = torch.min(time_bar[[u, v]]).item()
        if t_bar_min < time_keys_min:
            start_ind_min = 0
        elif t_bar_min > time_keys_max:
            # it means t_bar will always be larger, so there is no history for these nodes
            return s
        else:
            start_ind_min = self.time_keys.index(int(t_bar_min))

        expanded_time_bar = time_bar[[u, v]].view(1, 2).expand(N, -1).t().contiguous().view(2 * N, 1)
        # Adjust repeated time_bar to match the expanded shape
        adjusted_repeated_time_bar = time_bar.repeat(2, 1).view(2 * N, 1)
        # Now concatenate along dimension 1 (should work as both tensors are (168, 1))
        max_pairs = torch.max(torch.cat((expanded_time_bar, adjusted_repeated_time_bar), dim=1), dim=1)[0].view(2, N).long()

        # compute cond density for all pairs of u and some i, then of v and some i
        c1, c2 = 0, 0
        for c, j in enumerate([u, v]):  # range(i + 1, N):
            for i in range(N):
                if i == j:
                    continue
                # most recent timestamp of either u or v
                t_bar = max_pairs[c, i]
                c2 += 1

                if t_bar < time_keys_min:
                    start_ind = 0  # it means t_bar is beyond the history we kept, so use maximum period saved
                elif t_bar > time_keys_max:
                    continue  # it means t_bar is current event, so there is no history for this pair of nodes
                else:
                    # t_bar is somewhere in between time_keys_min and time_keys_min
                    start_ind = self.time_keys.index(t_bar, start_ind_min)

                indices.append((c, i))
                l_indices.append(start_ind)

        indices = np.array(indices)
        l_indices = np.array(l_indices)
        s[indices[:, 0], indices[:, 1]] = Lambda_sum[l_indices]

        return s
    
    # forward pass
    def forward(self,data):
        # opt is batch_update
        data[2] = data[2].float()
        data[4] = data[4].double()
        data[5] = data[5].double()
        u, v, k = data[0], data[1], data[3]
        time_delta_uv = data[2]
        time_bar = data[4]
        time_cur = data[5]
        event_types = k
        if self.with_attributes==True:
            data[6] = data[6].float()
            sim = data[6]
        B = len(u)
        assert len(event_types) == B, (len(event_types), B)
        N = self.N_nodes

        A_pred, Surv = None, None
        A_pred = self.A.new_zeros(B, N, N).fill_(0)
        Surv = self.A.new_zeros(B, N, N).fill_(0)

        if self.opt:
            embeddings1, embeddings2, node_degrees = [], [], []
            embeddings_non1, embeddings_non2 = [], []
        else:
            lambda_uv_t, lambda_uv_t_non_events = [], []

        assert torch.min(time_delta_uv) >= 0, ('events must be in chronological order', torch.min(time_delta_uv))

        time_mn = torch.from_numpy(np.array([0, 0, 0, 0])).float().view(1, 1, 4)
        time_sd = torch.from_numpy(np.array([50, 7, 15, 15])).float().view(1, 1, 4)
        time_delta_uv = (time_delta_uv - time_mn) / time_sd

        reg = []

        S_batch = []

        z_all = []

        u_all = u.data.cpu().numpy()
        v_all = v.data.cpu().numpy()


        for it, k in enumerate(event_types):
            # k = 0: association event (rare)
            # k = 1,2,3: communication event (frequent)

            u_it, v_it = u_all[it], v_all[it]
            z_prev = self.z if it == 0 else z_all[it - 1]
            if self.with_attributes==True:
                sim_it = sim[it]

            # 1. Compute intensity rate lambda based on node embeddings at previous time step (Eq. 1)
            if self.opt:
                # store node embeddings, compute lambda and S,A later based on the entire batch
                embeddings1.append(z_prev[u_it])
                embeddings2.append(z_prev[v_it])
            else:
                # accumulate intensity rate of events for this batch based on new embeddings
                lambda_uv_t.append(self.intensity_rate_lambda(z_prev[u_it], z_prev[v_it], torch.zeros(1).long() + k))
                # intensity_rate_lambda(z_u, z_v, k,n_hidden,psi,n_assoc_types,omega,edge_type=None)

            # 2. Update node embeddings
            if self.with_attributes==True:
                node_degree, z_new = self.update_node_embed_wa(z_prev, u_it, v_it, sim_it, time_delta_uv[it])
            else:
                node_degree, z_new = self.update_node_embed(z_prev, u_it, v_it, time_delta_uv[it])  # / 3600.)  # hours
            
            # update_node_embed(prev_embed, node1, node2, time_delta_uv, n_hidden,n_assoc_types, S, A, W_h, W_struct, W_rec, W_t)
            if self.opt:
                node_degrees.append(node_degree)


            # 3. Update S and A
            if not self.opt:
                # we can update S and A based on current pair of nodes even during test time,
                # because S, A are not used in further steps for this iteration
                self.update_S_A(u_it, v_it, k.item(), node_degree, lambda_uv_t[it])  #
                # update_S_A(A,S, u,v, k, node_degree, lambda_uv_t, N_nodes,n_assoc_types,node_degree_global)

            # update most recent degrees of nodes used to update S
            assert self.node_degree_global is not None
            for j in [u_it, v_it]:
                for rel in range(self.n_assoc_types):
                    self.node_degree_global[rel][j] = node_degree[j][rel]


            # Non events loss
            # this is not important for test time, but we still compute these losses for debugging purposes
            # get random nodes except for u_it, v_it
            # 4. compute lambda for sampled events that do not happen -> to compute survival probability in loss
            uv_others = self.rnd.choice(np.delete(np.arange(N), [u_it, v_it]), size= self.N_surv_samples * 2, replace=False)
                # assert len(np.unique(uv_others)) == len(uv_others), ('nodes must be unique', uv_others)
            for q in range(self.N_surv_samples):
                assert u_it != uv_others[q], (u_it, uv_others[q])
                assert v_it != uv_others[self.N_surv_samples + q], (v_it, uv_others[self.N_surv_samples + q])
                if self.opt:
                    embeddings_non1.extend([z_prev[u_it], z_prev[uv_others[self.N_surv_samples + q]]])
                    embeddings_non2.extend([z_prev[uv_others[q]], z_prev[v_it]])
                else:
                    for k_ in range(2):
                        lambda_uv_t_non_events.append(
                            self.intensity_rate_lambda(z_prev[u_it],
                                                        z_prev[uv_others[q]], torch.zeros(1).long() + k_))
                        lambda_uv_t_non_events.append(
                            self.intensity_rate_lambda(z_prev[uv_others[self.N_surv_samples + q]],
                                                        z_prev[v_it],
                                                        torch.zeros(1).long() + k_))


            # 5. compute conditional density for all possible pairs
            # here it's important NOT to use any information that the event between nodes u,v has happened
            # so, we use node embeddings of the previous time step: z_prev
            with torch.no_grad():
                z_cat = torch.cat((z_prev[u_it].detach().unsqueeze(0).expand(N, -1),
                                    z_prev[v_it].detach().unsqueeze(0).expand(N, -1)), dim=0)
                Lambda = self.intensity_rate_lambda(z_cat, z_prev.detach().repeat(2, 1),
                                                    torch.zeros(len(z_cat)).long() + k).detach()
                
                A_pred[it, u_it, :] = Lambda[:N]
                A_pred[it, v_it, :] = Lambda[N:]

                assert torch.sum(torch.isnan(A_pred[it])) == 0, (it, torch.sum(torch.isnan(A_pred[it])))
                # Compute the survival term for the current pair of nodes
                # we only need to compute the term for rows u_it and v_it in our matrix s to save time
                # because we will compute rank only for nodes u_it and v_it
                s1 = self.cond_density(time_bar[it], u_it, v_it)
                # cond_density(time_bar, u, v, N_nodes, Lambda_dict, time_keys)
                Surv[it, [u_it, v_it], :] = s1

                time_key = int(time_cur[it].item())
                idx = np.delete(np.arange(N), [u_it, v_it])  # nonevents for node u
                idx = np.concatenate((idx, idx + N))   # concat with nonevents for node v

                if len(self.time_keys) >= len(self.Lambda_dict):
                    # shift in time (remove the oldest record)
                    time_keys = np.array(self.time_keys)
                    time_keys[:-1] = time_keys[1:]
                    self.time_keys = list(time_keys[:-1])  # remove last
                    self.Lambda_dict[:-1] = self.Lambda_dict.clone()[1:]
                    self.Lambda_dict[-1] = 0

                self.Lambda_dict[len(self.time_keys)] = Lambda[idx].sum().detach()  # total intensity of non events for the current time step
                self.time_keys.append(time_key)

            # Once we made predictions for the training and test sample, we can update node embeddings
            z_all.append(z_new)
            # update S

            self.A = self.S
            S_batch.append(self.S.data.cpu().numpy())

            self.t_p += 1

        self.z = z_new  # update node embeddings

        # Batch update
        if self.opt:
            lambda_uv_t = self.intensity_rate_lambda(torch.stack(embeddings1, dim=0),
                                                        torch.stack(embeddings2, dim=0), event_types)
            non_events = len(embeddings_non1)
            n_types = 2
            lambda_uv_t_non_events = torch.zeros(non_events * n_types)
            embeddings_non1 = torch.stack(embeddings_non1, dim=0)
            embeddings_non2 = torch.stack(embeddings_non2, dim=0)
            idx = None
            empty_t = torch.zeros(non_events, dtype=torch.long)
            types_lst = torch.arange(n_types)
            for k in types_lst:
                if idx is None:
                    idx = np.arange(non_events)
                else:
                    idx += non_events
                lambda_uv_t_non_events[idx] = self.intensity_rate_lambda(embeddings_non1, embeddings_non2, empty_t + k)

            # update only once per batch
            for it, k in enumerate(event_types):
                u_it, v_it = u_all[it], v_all[it]
                self.update_S_A(u_it, v_it, k.item(), node_degrees[it], lambda_uv_t[it].item())
                # update_S_A(A,S, u, v, k, node_degree, lambda_uv_t, N_nodes,n_assoc_types,node_degree_global)

        else:
            lambda_uv_t = torch.cat(lambda_uv_t)
            lambda_uv_t_non_events = torch.cat(lambda_uv_t_non_events)


        if len(reg) > 1:
            reg = [torch.stack(reg).mean()]

        return lambda_uv_t, lambda_uv_t_non_events / self.N_surv_samples, [A_pred, Surv], reg
