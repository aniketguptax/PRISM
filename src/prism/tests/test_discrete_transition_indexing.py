from prism.reconstruction.one_step_merge import OneStepGreedyMerge
from prism.representations.discrete import LastK


def test_transition_counts_match_defined_m_t_and_m_t_plus_1_indices() -> None:
    x_train = [0, 0, 0, 0, 0]
    rep = LastK(k=2)
    model = OneStepGreedyMerge(eps=0.0).fit(x_train, rep, seed=0)

    expected = 3
    observed = sum(model.sa_counts.values())
    assert observed == expected


def test_occupancy_uses_all_defined_m_t_not_only_transition_positions() -> None:
    x_train = [0, 1, 0, 1, 0]
    rep = LastK(k=2)
    model = OneStepGreedyMerge(eps=0.0).fit(x_train, rep, seed=0)

    s01 = model.rep_to_state[(0, 1)]
    s10 = model.rep_to_state[(1, 0)]
    assert model.pi[s01] == 0.5
    assert model.pi[s10] == 0.5
