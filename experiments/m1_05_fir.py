from pathlib import Path

import numpy as np

from anc.filters.basic import fir_filter
from anc.signals.signal import Signal

def main() -> None:
    sampling_rate = 16_000

    x = Signal(
        samples=np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0]
        ),
        sampling_rate=sampling_rate,
        metadata={
            "signal_type": "test_sequence",
        },
    )

    h = np.array(
        [1.0, 0.5, -0.25]
    )

    y= fir_filter(x, h)

    print("Input x[n]:")
    print(x.samples)

    print("\nFIR h[n]:")
    print(h)

    print("\nOutput y[n]:")
    print(y.samples)

    output_dir = Path("results/m1_05_fir")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_dir / "input.npy",
        x.samples,
    )

    np.save(
        output_dir / "coefficients.npy",
        h,
    )

    np.save(
        output_dir / "output.npy",
        y.samples,
    )


if __name__ == "__main__":
    main()
