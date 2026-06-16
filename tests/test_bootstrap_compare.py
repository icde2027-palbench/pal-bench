from scripts.evidence_chain_dossier.bootstrap_compare import _paired_bootstrap


def test_paired_bootstrap_reports_small_p_when_ci_excludes_zero() -> None:
    result = _paired_bootstrap(
        [2.0, 2.2, 1.8, 2.1, 1.9],
        [1.0, 1.1, 0.9, 1.0, 1.2],
        n=1000,
        seed=7,
    )

    assert result["ci_low"] > 0
    assert result["paired_p_two_sided_bootstrap"] == 0.0


def test_paired_bootstrap_reports_non_significant_p_when_centered_on_zero() -> None:
    result = _paired_bootstrap(
        [2.0, 1.0, 2.0, 1.0],
        [1.0, 2.0, 1.0, 2.0],
        n=1000,
        seed=7,
    )

    assert result["mean_delta"] == 0.0
    assert result["paired_p_two_sided_bootstrap"] == 1.0
