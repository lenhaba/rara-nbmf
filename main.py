# %%
import dimod
import gurobipy as gp
import numpy as np
from dwave.embedding import embed_qubo, unembed_sampleset
from dwave.embedding.chimera import find_clique_embedding
from dwave.system import DWaveSampler
from PIL import Image
from tqdm.auto import tqdm
from tqdm.notebook import trange


# %%
def nlssubprob(V, W, H_init, tol, maxiter, doProject=False):
    H = H_init
    WtV = np.dot(W.T, V)
    WtW = np.dot(W.T, W)

    alpha = 0.1
    beta = 0.1

    for _ in range(maxiter):
        grad = np.dot(WtW, H) - WtV
        projgrad = np.linalg.norm(np.where(((grad < 0) | (H > 0)), grad, 0), 2)
        if projgrad < tol:
            break

        for inner_iter in range(20):
            Hn = np.where((H - alpha * grad) > 0, (H - alpha * grad), 0)
            if doProject:
                Hn = np.where(Hn < 1, Hn, 1)

            d = Hn - H
            gradd = sum(sum(np.multiply(grad, d)))
            dQd = sum(sum(np.multiply(np.dot(WtW, d), d)))
            suff_decr = (0.99 * gradd + 0.5 * dQd) < 0
            if inner_iter == 0:
                decr_alpha = not suff_decr
                Hp = H

            if decr_alpha:
                if suff_decr:
                    H = Hn
                    break
                else:
                    alpha = alpha * beta
            else:
                if (not suff_decr) or np.all(Hp == Hn):
                    H = Hp
                    break
                else:
                    alpha = alpha / beta
                    Hp = Hn

    return H


def make_all_quboMatrix(V, W):
    n = V.shape[1]
    WtW = W.T @ W
    VtW = V.T @ W

    allQUBO = {}
    for i_n in range(n):
        allQUBO[i_n] = -2 * np.diag(VtW[i_n], k=0) + WtW

    return allQUBO


def embed_initial(var_x, embedding):
    initial = {}
    for k, chain in embedding.items():
        for qubit in chain:
            initial[qubit] = var_x[k]
    return initial


def Dwave_optimimize(qubo, sampler, num_reads, feature=None, **kwargs):

    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)
    embedding = global_embedding

    qubo_embeded = embed_qubo(qubo, embedding, sampler.adjacency)

    if feature == "Forward Annealing":

        raw_sampleset = sampler.sample_qubo(
            qubo_embeded, num_reads=num_reads, annealing_time=20
        )
        forward_unembeded_sampleset = unembed_sampleset(raw_sampleset, embedding, bqm)

        var_x = forward_unembeded_sampleset.first.sample
        for v in var_x.values():
            assert (v == 0) or (v == 1)

    if feature == "Reverse Annealing":
        reverse_schedule = [[0.0, 1.0], [10.0, 0.55], [20.0, 0.55], [30.0, 1.0]]
        initial = embed_initial(kwargs["initial_state"], embedding)
        reverse_anneal_params = dict(
            anneal_schedule=reverse_schedule,
            initial_state=initial,
            reinitialize_state=True,
        )
        reverse_sampleset = sampler.sample_qubo(
            qubo_embeded, num_reads=num_reads, **reverse_anneal_params
        )
        reverse_unembeded_sampleset = unembed_sampleset(
            reverse_sampleset, embedding, bqm
        )
        var_x = reverse_unembeded_sampleset.first.sample

    return var_x


def solve_Gurobi(V_vec, W):

    k = W.shape[1]
    m = W.shape[0]
    k = len(W[0])

    model = gp.Model()
    model.Params.LogToConsole = 0
    x = {i: model.addVar(vtype=gp.GRB.BINARY, name=f"x({i})") for i in range(k)}
    f = sum(
        (V_vec[i_m] - sum(W[i_m][i_k] * x[i_k] for i_k in range(k))) ** 2
        for i_m in range(m)
    )
    model.setObjective(f, sense=gp.GRB.MINIMIZE)
    model.optimize()
    if model.Status == gp.GRB.OPTIMAL:
        var_x = {i: round(x[i].X) for i in range(k)}
    else:
        print("Error: Infeasible solution", file=sys.stderr)
        raise

    return var_x


def optimize_H(V, W, Hinit, solver, num_reads=1000, sampler=None):

    n = V.shape[1]
    H = []

    if solver == "PGDRA":
        H_PGD = nlssubprob(V, W, Hinit, tolH, 100, doProject=True)
        H_PGD = np.where(H_PGD < 0.5, 0, 1)

    if solver == "Gurobi":
        for i_n in trange(n, leave=False):
            spin_x = solve_Gurobi(V.T[i_n], W)
            H.append(list(spin_x.values()))

    else:
        allQUBO = make_all_quboMatrix(V, W)
        for i_n in trange(n, leave=False):
            if solver == "FA":
                var_x = Dwave_optimimize(
                    allQUBO[i_n],
                    sampler,
                    num_reads=1000,
                    feature="Forward Annealing",
                    global_embedding=global_embedding,
                )
            elif solver == "QARA":
                var_x_fa = Dwave_optimimize(
                    allQUBO[i_n],
                    sampler,
                    num_reads=1000,
                    feature="Forward Annealing",
                    global_embedding=global_embedding,
                )
                var_x = Dwave_optimimize(
                    allQUBO[i_n],
                    sampler,
                    num_reads=240,
                    feature="Reverse Annealing",
                    initial_state=var_x_fa,
                    global_embedding=global_embedding,
                )
            elif solver == "PGDRA":
                var_x_pgd = H_PGD.T[i_n]
                var_x = Dwave_optimimize(
                    allQUBO[i_n],
                    sampler,
                    num_reads=240,
                    feature="Reverse Annealing",
                    initial_state=var_x_pgd,
                    global_embedding=global_embedding,
                )
            else:
                print("Error: Invalid sover name " + str(solver), file=sys.stderr)
                sys.exit(1)

            H.append(list(var_x.values()))

    H = np.array(H).T
    return H


# %%

# Load data
num_images = 200
width, height = (19, 19)
num_feature = 7 * 5

V = []
for i in range(1, num_images + 1):
    img = Image.open("data/face/face" + f"{i:05}" + ".pgm")
    img_matrix = np.array(img)
    V.append(img_matrix.flatten())
V = np.array(V).T

m = width * height
n = num_images
k = num_feature
assert k < n * m / (n + m)


# Configurations
H = np.random.randint(0, 2, (k, n))
W = np.random.uniform(0, 1, (m, k))


tolW = 1
tolH = 1
train_maxIter = 10
grad_maxIter = 100


# %%

# PGD

for i in tqdm(range(1, train_maxIter + 1)):
    W = nlssubprob(V.T, H.T, W.T, tolW, grad_maxIter).T
    H = nlssubprob(V, W, H, tolH, grad_maxIter, doProject=True)
    H = np.where(H < 0.5, 0, 1)

# %%

# Gurobi


for i in tqdm(range(1, train_maxIter + 1)):
    W = nlssubprob(V.T, H.T, W.T, tolW, grad_maxIter).T
    H = optimize_H(V, W, H, solver="Gurobi")

# %%

# QA -> RA

endpoint = "https://cloud.dwavesys.com/sapi"
token = ""
solver = "DW_2000Q_6"
sampler = DWaveSampler(endpoint=endpoint, token=token, solver=solver)
global_embedding = find_clique_embedding(k, 16, n=None, t=None)


for i in tqdm(range(1, train_maxIter + 1)):

    W = nlssubprob(V.T, H.T, W.T, tolW, grad_maxIter).T
    H = optimize_H(
        V, W, H, solver="FA", sampler=sampler, global_embedding=global_embedding
    )

# %%

# QA -> RA

for i in tqdm(range(1, train_maxIter + 1)):
    W = nlssubprob(V.T, H.T, W.T, tolW, grad_maxIter).T
    H = optimize_H(
        V, W, H, solver="QARA", sampler=sampler, global_embedding=global_embedding
    )

# %%

# PGD -> RA

for i in tqdm(range(1, train_maxIter + 1)):
    W = nlssubprob(V.T, H.T, W.T, tolW, grad_maxIter).T
    H = optimize_H(
        V, W, H, solver="PGDRA", sampler=sampler, global_embedding=global_embedding
    )
