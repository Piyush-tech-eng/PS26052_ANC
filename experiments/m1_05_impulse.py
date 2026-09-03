from pathlib import Path

import numpy as np

from anc.signals.generators import generate_impulse


def main() -> None:
    sampling_rate = 16_000
    duration = 0.01
    impulse_index = 0
    amplitude = 1.0

    signal = generate_impulse(
        sampling_rate=sampling_rate,
        duration=duration,
        index=impulse_index,
        amplitude=amplitude,
    )

    output_dir = Path("results/m1_05_impulse")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_dir / "signal.npy",
        signal.samples,
    )

    print("Impulse signal generated.")
    print(f"Samples: {signal.num_samples}")
    print(f"Sampling rate: {signal.sampling_rate} Hz")
    print(f"Impulse index: {impulse_index}")
    print(f"Impulse amplitude: {amplitude}")
    print(f"Maximum sample: {np.max(signal.samples)}")

if __name__ == "__main__":
    main()