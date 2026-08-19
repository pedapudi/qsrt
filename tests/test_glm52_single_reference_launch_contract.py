from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CACHE = (
    "runtime-cache/"
    "glm52-per-expert-exl3-without-fused-staging-dense-triton-bf16-reference-kld"
)


def test_single_reference_launchers_reuse_the_registered_numerical_runtime() -> None:
    scripts = (
        "experiments/run_glm52_frozen_expert_subset_single_reference_on_kossel.sh",
        "experiments/run_glm52_layer63_retained_down_refit_single_reference_on_kossel.sh",
    )
    for relative_path in scripts:
        source = (REPOSITORY_ROOT / relative_path).read_text()
        assert CANONICAL_CACHE in source
        assert 'runtime_cache="${experiment_root}/runtime-cache/${result_name}"' not in source


def test_layer52_composition_opens_the_long_reference_only_after_selection() -> None:
    source = (
        REPOSITORY_ROOT
        / "experiments/build_and_screen_glm52_layer52_model_kld_selected_"
        "down_recovery_composition_on_kossel.sh"
    ).read_text()

    selection_call = source.index(
        "run_glm52_complete_panel_public_reference_screen_on_kossel.sh"
    )
    selection_decision = source.index("first_mean < 0.0 and second_mean < 0.0")
    independent_call = source.index(
        "run_glm52_frozen_expert_subset_single_reference_on_kossel.sh"
    )
    assert selection_call < selection_decision < independent_call
    assert "test ! -e \"${decision}\"" in source
    assert "2,048-token development reference" in source
