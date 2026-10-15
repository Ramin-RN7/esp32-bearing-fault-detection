# /model_rt_gmm.py — A8: scale -> PCA -> GMM (diag/spherical), ulab-minimal
import ujson
from ulab import numpy as np

LN_2PI = 1.8378770664093453  # ln(2*pi)

def jload(p):
    with open(p) as f:
        return ujson.load(f)

def _to_np1d(lst):
    return np.array([float(x) for x in lst])

def _pca_project(v, comps, mean):
    if mean is not None:
        v = v - _to_np1d(mean)
    K = len(comps)
    D = len(comps[0])
    y = np.zeros(K)
    for k in range(K):
        s = 0.0
        row = comps[k]
        for d in range(D):
            s += v[d] * row[d]
        y[k] = s
    return y

def preprocess_vec(x25, pre):
    v = _to_np1d(x25)

    sc = pre.get("scale")
    if sc:
        if sc.get("type") == "minmax":
            v = v * _to_np1d(sc["scale"]) + _to_np1d(sc["min"])
        elif sc.get("type") == "z":
            v = (v - _to_np1d(sc["mean"])) / (_to_np1d(sc["scale"]) + 1e-8)

    pca = pre.get("pca")
    if pca:
        comps = pca["components"]
        mean  = pca.get("mean")
        v = _pca_project(v, comps, mean)
    return v

def _log_gauss_diag(x, mu, var):
    var = var + 1e-8
    D = len(x)
    diff = x - mu
    t1 = -0.5 * np.sum((diff * diff) / var)
    t2 = -0.5 * (np.sum(np.log(var)) + D * LN_2PI)
    return t1 + t2

def gmm_loglik(xk, gmm):
    covt = gmm.get("covariance_type", "diag")
    w = _to_np1d(gmm["weights"])
    means = np.array([_to_np1d(r) for r in gmm["means"]])

    covs_raw = gmm["covariances"]
    if isinstance(covs_raw[0], list):  # diag
        covs = np.array([_to_np1d(r) for r in covs_raw])
    else:                              # spherical
        K = len(means[0]); M = len(covs_raw)
        covs = np.zeros((M, K))
        for i in range(M):
            val = float(covs_raw[i])
            for k in range(K):
                covs[i, k] = val

    if covt not in ("diag", "spherical"):
        raise ValueError("GMM cov type not supported: %s" % covt)

    logw = np.log(w + 1e-12)
    M = len(means)
    logs = np.zeros(M)
    for k in range(M):
        logs[k] = logw[k] + _log_gauss_diag(xk, means[k], covs[k])
    m = float(np.max(logs))
    return m + np.log(np.sum(np.exp(logs - m)) + 1e-12)

def predict_gmm(feat25):
    pre = jload("/models/A8_GMM/preprocess.json")
    gmm = jload("/models/A8_GMM/gmm.json")
    xk = preprocess_vec(feat25, pre)   # -> PCA-K (should be 4)
    ll = gmm_loglik(xk, gmm)
    thr = float(gmm.get("thr", 0.0))
    y = 1 if ll >= thr else 0
    return int(y), float(ll), float(thr)