import torch
import numpy as np

def MAR(A_pred, u, v, k, Survival_term):
    '''Computes mean average ranking for a batch of events'''
    ranks = []
    hits_10 = []
    N = len(A_pred)
    Survival_term = torch.exp(-Survival_term)
    A_pred *= Survival_term
    assert torch.sum(torch.isnan(A_pred)) == 0, (torch.sum(torch.isnan(A_pred)), Survival_term.min(), Survival_term.max())

    A_pred = A_pred.data.cpu().numpy()


    assert N == len(u) == len(v) == len(k), (N, len(u), len(v), len(k))
    for b in range(N):
        u_it, v_it = u[b].item(), v[b].item()
        assert u_it != v_it, (u_it, v_it, k[b])
        A = A_pred[b].squeeze()
        # remove same node
        idx1 = list(np.argsort(A[u_it])[::-1])
        idx1.remove(u_it)
        idx2 = list(np.argsort(A[v_it])[::-1])
        idx2.remove(v_it)
        rank1 = np.where(np.array(idx1) == v_it) # get nodes most likely connected to u[b] and find out the rank of v[b] among those nodes
        rank2 = np.where(np.array(idx2) == u_it)  # get nodes most likely connected to v[b] and find out the rank of u[b] among those nodes
        assert len(rank1) == len(rank2) == 1, (len(rank1), len(rank2))
        hits_10.append(np.mean([float(rank1[0] <= 9), float(rank2[0] <= 9)]))
        rank = np.mean([rank1[0], rank2[0]])
        assert isinstance(rank, np.float64), (rank, rank1, rank2, u_it, v_it, idx1, idx2)
        ranks.append(rank)
    return ranks, hits_10

def relative_stability(F, St, St_next, Vt, Vt_next):
    # Flatten the adjacency matrices if they are 3-dimensional
    if len(St.shape) == 3:
        St = St.reshape(St.shape[0], -1)
        St_next = St_next.reshape(St_next.shape[0], -1)
    
    # Compute the differences in embeddings
    embedding_diff = np.linalg.norm(Vt_next - Vt, 'fro') / np.linalg.norm(Vt, 'fro')
    
    # Compute the differences in adjacency matrices
    adjacency_diff = np.linalg.norm(St_next - St, 'fro') / np.linalg.norm(St, 'fro')
    
    # Relative stability
    S_r = embedding_diff / adjacency_diff
    
    return S_r


def stability_constant(F, embeddings_list, adj_matrices_list):
    max_relative_stability_diff = 0
    for i in range(len(embeddings_list) - 1):
        Vt = embeddings_list[i]
        Vt_next = embeddings_list[i + 1]
        St = adj_matrices_list[i]
        St_next = adj_matrices_list[i + 1]
        S_r_t = relative_stability(F, St, St_next, Vt, Vt_next)
        for j in range(i + 1, len(embeddings_list) - 1):
            Vt_prime = embeddings_list[j]
            Vt_next_prime = embeddings_list[j + 1]
            St_prime = adj_matrices_list[j]
            St_next_prime = adj_matrices_list[j + 1]
            S_r_t_prime = relative_stability(F, St_prime, St_next_prime, Vt_prime, Vt_next_prime)
            stability_diff = abs(S_r_t - S_r_t_prime)
            if stability_diff > max_relative_stability_diff:
                max_relative_stability_diff = stability_diff
    return max_relative_stability_diff


def initialize_state(dataset, model, node_embeddings, keepS=False):
    Adj_all = dataset.get_Adjacency()[0]

    if not isinstance(Adj_all, list):
        Adj_all = [Adj_all]

    # Ensure it is a list of adjacency matrices
    if Adj_all[0].ndim == 1 and Adj_all[0].size == dataset.N_nodes:
        # If it's a vector that should be a diagonal of a matrix
        Adj_matrix = np.zeros((dataset.N_nodes, dataset.N_nodes))
        np.fill_diagonal(Adj_matrix, Adj_all[0])
        Adj_all[0] = Adj_matrix[:, :, None]  # Convert to 3D by adding a new axis

    node_degree_global = []
    for rel, A in enumerate(Adj_all):
        node_degree_global.append(np.sum(A, axis=1))  # Sum over columns to get degrees

    time_bar = np.zeros((dataset.N_nodes, 1)) + dataset.FIRST_DATE.timestamp()

    model.initialize(node_embeddings=node_embeddings, A_initial=Adj_all[0], keepS=keepS)

    return time_bar, node_degree_global