"""Label logprobs -> Jev answers. The distribution is a softmax over the requested labels only."""
import math


def softmax(logprobs, T=1.0):
    peak = max(logprobs)
    if not math.isfinite(peak):
        raise ValueError("no finite label logprob from the backend")
    w = [math.exp((x - peak) / T) for x in logprobs]
    s = math.fsum(w)
    return [x / s for x in w]


def confidence(p):
    """1 - H(p)/log K, clamped: 0 for uniform, 1 for a point mass."""
    h = -math.fsum(x * math.log(x) for x in p if x > 0)
    return min(1.0, max(0.0, 1 - h / math.log(len(p))))


def answer(question, keys, probs):
    dist = dict(zip(keys, probs))
    typ = question["type"]
    if typ == "noul":
        return {"type": typ, "noul": dist["true"]}
    if typ == "score":
        return {"type": typ, "score": math.fsum(i * p for i, p in enumerate(probs)), "probabilities": dist,
                "legend": {str(i): v for i, v in enumerate(question["criteria"])}, "confidence": confidence(probs)}
    return {"type": typ, "choice": max(dist, key=dist.__getitem__), "probabilities": dist, "confidence": confidence(probs)}
