"""M2.01 — Load and validate the frozen Module 1 handoff."""

from pathlib import Path

from anc.statistics import (
    load_module1_handoff,
    save_m2_input_summary,
)


def main() -> None:
    """Load canonical Module 1 inputs and record their M2.01 contract."""

    project_root = Path(__file__).resolve().parents[1]
    handoff_directory = project_root / "results" / "m1_handoff"
    summary_path = (
        project_root
        / "results"
        / "m2_01_load_handoff"
        / "m2_input_summary.json"
    )

    handoff = load_module1_handoff(handoff_directory)
    save_m2_input_summary(handoff, summary_path)

    print("M2.01 — Load Module 1 Handoff")
    print("--------------------------------")
    print(f"Sampling rate: {handoff.sampling_rate_hz} Hz")
    print(f"Number of samples: {handoff.num_samples}")
    print(
        "Reference x[n]: "
        f"shape={handoff.reference.shape}, "
        f"dtype={handoff.reference.dtype}"
    )
    print(
        "Desired d[n]: "
        f"shape={handoff.desired.shape}, "
        f"dtype={handoff.desired.dtype}"
    )
    print("Validation: equal lengths, finite samples, valid sampling rate")
    print(f"Input summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
