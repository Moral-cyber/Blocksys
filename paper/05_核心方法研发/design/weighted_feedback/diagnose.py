"""Finite NumPy accounting fixtures for time-varying weighted error feedback.

Synthetic deltas are frozen independently of both compressor trajectories.
No data, model, learning metric, validation set, or test set is used.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
TOLERANCE = 2e-12


def topk(x, k):
    indices = np.argsort(-np.abs(x), kind="stable")[:k]
    sent = np.zeros_like(x)
    sent[indices] = x[indices]
    return sent


def digest(x):
    x = np.ascontiguousarray(x)
    return hashlib.sha256(str(x.dtype).encode() + str(x.shape).encode() + x.tobytes()).hexdigest()


def simulate(deltas, active, weights, k):
    """Compare local-unit residuals with server-contribution-unit debts."""
    rounds, clients, dimension = deltas.shape
    r = np.zeros((clients, dimension), dtype=np.float64)
    e = np.zeros_like(r)
    target = np.zeros(dimension, dtype=np.float64)
    local_uploads = np.zeros_like(target)
    global_uploads = np.zeros_like(target)
    last_a = np.zeros(clients, dtype=np.float64)
    drift = np.zeros_like(target)
    rows = []
    max_inactive_change = 0.
    max_round_upload_difference = 0.
    max_scaled_state_difference = 0.
    max_local_identity_error = 0.
    max_global_identity_error = 0.
    for t in range(rounds):
        before_r, before_e = r.copy(), e.copy()
        weighted_delta = np.zeros(dimension, dtype=np.float64)
        local_round = np.zeros_like(weighted_delta)
        global_round = np.zeros_like(weighted_delta)
        for i in active[t]:
            a = weights[t, i]
            assert a > 0
            # Summation by parts: changing a revalues an existing local debt.
            if last_a[i] != 0:
                drift += (last_a[i] - a) * r[i]
            s = topk(deltas[t, i] + r[i], k)
            r[i] = deltas[t, i] + r[i] - s
            v = a * deltas[t, i] + e[i]
            q = topk(v, k)
            e[i] = v - q
            weighted_delta += a * deltas[t, i]
            local_round += a * s
            global_round += q  # already weighted; do not multiply a again
            last_a[i] = a
        inactive = np.setdiff1d(np.arange(clients), active[t])
        if len(inactive):
            max_inactive_change = max(max_inactive_change,
                float(np.max(np.abs(r[inactive] - before_r[inactive]))),
                float(np.max(np.abs(e[inactive] - before_e[inactive]))))
        target += weighted_delta
        local_uploads += local_round
        global_uploads += global_round
        local_debt = target - local_uploads
        global_debt = target - global_uploads
        final_weighted_r = np.sum(last_a[:, None] * r, axis=0)
        local_identity_error = float(np.max(np.abs(local_debt - final_weighted_r - drift)))
        global_identity_error = float(np.max(np.abs(global_debt - np.sum(e, axis=0))))
        max_local_identity_error = max(max_local_identity_error, local_identity_error)
        max_global_identity_error = max(max_global_identity_error, global_identity_error)
        max_round_upload_difference = max(max_round_upload_difference, float(np.max(np.abs(local_round-global_round))))
        max_scaled_state_difference = max(max_scaled_state_difference, float(np.max(np.abs(e-last_a[:, None]*r))))
        rows.append({"round":t+1,"active":active[t].tolist(),"weights":weights[t].tolist(),
            "uncompressed_weighted_delta":weighted_delta.tolist(),"local_weighted_upload":local_round.tolist(),
            "global_unit_upload":global_round.tolist(),"cumulative_local_debt":local_debt.tolist(),
            "last_active_weighted_local_residual":final_weighted_r.tolist(),"weight_change_drift":drift.tolist(),
            "cumulative_global_debt":global_debt.tolist(),"sum_global_unit_residual":np.sum(e,axis=0).tolist(),
            "local_summation_by_parts_error":local_identity_error,"global_telescope_error":global_identity_error})
    assert max_inactive_change == 0
    assert max_local_identity_error < TOLERANCE
    assert max_global_identity_error < TOLERANCE
    return {"rounds":rounds,"clients":clients,"dimension":dimension,"K":k,
        "max_inactive_state_change":max_inactive_change,
        "max_local_summation_by_parts_error":max_local_identity_error,
        "max_global_telescope_error":max_global_identity_error,
        "max_round_upload_difference_between_methods":max_round_upload_difference,
        "max_e_minus_last_a_r":max_scaled_state_difference,
        "final_target":target.tolist(),"final_local_uploads":local_uploads.tolist(),
        "final_global_unit_uploads":global_uploads.tolist(),
        "final_local_debt":(target-local_uploads).tolist(),
        "final_last_active_weighted_r":np.sum(last_a[:,None]*r,axis=0).tolist(),
        "final_local_weight_change_drift":drift.tolist(),
        "final_global_debt":(target-global_uploads).tolist(),"final_sum_e":np.sum(e,axis=0).tolist(),
        "naive_local_final_residual_identity_gap_norm":float(np.linalg.norm(target-local_uploads-np.sum(last_a[:,None]*r,axis=0))),
        "final_local_residuals":r.tolist(),"final_global_unit_residuals":e.tolist(),"rows":rows}


def normalized_weights(sizes, active):
    weights=np.zeros((len(active),len(sizes)),dtype=np.float64)
    for t,ids in enumerate(active):
        weights[t,ids]=sizes[ids]/np.sum(sizes[ids])
    return weights


def run():
    # A two-client external-weight algebra example cannot be generated by two
    # fixed-size clients under active normalization (.1 -> 1/91 requires a third).
    hand_delta=np.zeros((2,2,2),dtype=np.float64)
    hand_delta[0,0]=[1.,.9]
    hand_active=np.array([[0,1],[0,1]],dtype=np.int64)
    hand_weights=np.array([[.1,.9],[1/91,90/91]],dtype=np.float64)
    hand=simulate(hand_delta,hand_active,hand_weights,1)
    assert np.max(np.abs(hand["final_local_residuals"]))==0
    expected=.9*(.1-1/91)
    assert np.allclose(hand["final_local_debt"],[0,expected],rtol=0,atol=TOLERANCE)
    assert np.max(np.abs(hand["final_global_debt"]))<TOLERANCE
    assert np.max(np.abs(hand["final_global_unit_residuals"]))==0
    legal_delta=np.zeros((2,3,2),dtype=np.float64)
    legal_delta[0,0]=[1.,.9]
    legal_sizes=np.array([1,9,90],dtype=np.int64)
    legal_active=np.array([[0,1],[0,2]],dtype=np.int64)
    legal=simulate(legal_delta,legal_active,normalized_weights(legal_sizes,legal_active),1)
    assert np.allclose(legal["final_local_debt"],hand["final_local_debt"],rtol=0,atol=TOLERANCE)

    rounds,clients,dimension,k,participants=60,20,13,3,5
    sizes=np.array([31,47,79,103,149,211,293,431,607,887,1201,1601,2203,2909,4001,5501,7603,10007,15013,21011],dtype=np.int64)
    schedule_rng=np.random.default_rng(20261171)
    active=np.array([np.sort(schedule_rng.choice(clients,size=participants,replace=False)) for _ in range(rounds)])
    delta_rng=np.random.default_rng(20261172)
    client_offsets=delta_rng.normal(0,.6,size=(clients,dimension))
    deltas=(delta_rng.normal(size=(rounds,clients,dimension))+client_offsets[None,:,:])/(1+.025*np.arange(rounds)[:,None,None])
    weights=normalized_weights(sizes,active)
    variable=simulate(deltas,active,weights,k)
    equal_sizes=np.ones(clients,dtype=np.int64)
    equal_weights=normalized_weights(equal_sizes,active)
    equal=simulate(deltas,active,equal_weights,k)
    full_active=np.tile(np.arange(clients),(rounds,1))
    full_weights=normalized_weights(sizes,full_active)
    full=simulate(deltas,full_active,full_weights,k)
    for name,result in (("equal_sizes_uniform_fixed_participation",equal),("heterogeneous_sizes_constant_full_participation",full)):
        assert result["max_round_upload_difference_between_methods"]<TOLERANCE,name
        assert result["max_e_minus_last_a_r"]<TOLERANCE,name
        assert result["naive_local_final_residual_identity_gap_norm"]<TOLERANCE,name
    assert variable["naive_local_final_residual_identity_gap_norm"]>1e-3
    assert variable["max_round_upload_difference_between_methods"]>1e-3
    report={"status":"PASS","scope":"finite synthetic algebra and raw numerical accounting; not learning results",
        "numpy_version":np.__version__,"float_dtype":"float64","tolerance":TOLERANCE,
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "identities":{
            "local":"D_local = sum_i(last_active_a_i * r_i_final) + sum_i,sum_consecutive_activations((a_prev-a_next)*r_before_next)",
            "global":"D_global = sum_i e_i_final (zero initial debt)",
            "constant_weight_equivalence":"If each client's active a_i is constant and positive, e_i=a_i*r_i and q_i=a_i*s_i in exact arithmetic, even with inactive rounds",
            "target":"D is cumulative actual uncompressed active-normalized weighted deltas minus cumulative transmitted server updates, not a learning gradient truth"},
        "two_client_external_weight_hand_example":{
            "limitation":"The coefficients .1 then 1/91 cannot arise from only two fixed-n clients under active normalization; this is an externally prescribed coefficient algebra example",
            "weights":hand_weights.tolist(),"deltas":hand_delta.tolist(),"expected_local_untracked_debt_second_coordinate":expected,"result":hand},
        "three_client_fixed_n_legal_hand_example":{"sizes":legal_sizes.tolist(),"active":legal_active.tolist(),
            "explanation":"Both active sets have size two; they can occur under uniform size-two sampling. This chosen hand schedule is an algebra fixture, not a random sample inference.","result":legal},
        "frozen_20_client_inputs":{"sampling":"uniform without replacement, fixed five of twenty per round; sorted only for canonical processing",
            "schedule_seed":20261171,"delta_seed":20261172,"sizes":sizes.tolist(),"active":active.tolist(),
            "deltas":deltas.tolist(),"delta_generation":"predefined independent client offsets plus round noise; no method/model feedback",
            "sha256":{"sizes":digest(sizes),"active":digest(active),"deltas":digest(deltas),"variable_weights":digest(weights)}},
        "variable_sizes_uniform_fixed_size_sampling":variable,
        "equal_sizes_same_uniform_fixed_size_schedule":equal,
        "heterogeneous_sizes_constant_full_participation":full,
        "limitations":["No real dataset, label heterogeneity, training, accuracy, convergence or novelty claim",
            "Changing normalization also changes the uncompressed sampled objective; accounting correction does not remove sampling bias",
            "Global-unit EF is a well-defined candidate convention, not proof it improves performance or is novel",
            "Each active client must know its a_i before compression; server sums q_i without weighting again",
            "Inactive state is unchanged; all active packets are assumed delivered and committed",
            "The same frozen delta stream is deliberately used for both methods; real closed-loop deltas would diverge"]}
    path=HERE/"diagnose.json"
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    compact={"status":report["status"],"two_client_legal_under_fixed_n_sampling":False,
        "hand_local_debt":hand["final_local_debt"],"hand_local_final_residual":hand["final_last_active_weighted_r"],
        "hand_global_debt":hand["final_global_debt"],
        "variable_weight_naive_identity_gap_norm":variable["naive_local_final_residual_identity_gap_norm"],
        "variable_weight_correct_local_identity_error":variable["max_local_summation_by_parts_error"],
        "global_identity_error":variable["max_global_telescope_error"],
        "equal_weight_method_equivalence_error":equal["max_round_upload_difference_between_methods"],
        "constant_full_participation_equivalence_error":full["max_round_upload_difference_between_methods"],
        "output":str(path)}
    print(json.dumps(compact,indent=2))


if __name__=="__main__":
    run()
