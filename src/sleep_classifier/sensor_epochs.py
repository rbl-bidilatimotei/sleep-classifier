from numbers import Real

import numpy as np
import pandas as pd


class SessionEpochClock:
    """Schedule completed epochs from a declared session clock and sensor watermarks."""

    def __init__(self, start_seconds: float = 0.0) -> None:
        """Declare the first epoch start independently of sleep-label timestamps.

        Args:
            start_seconds (float): Session-relative origin for thirty-second intervals.
        """
        if (isinstance(start_seconds, (bool, np.bool_)) or not isinstance(start_seconds, Real)
                or not np.isfinite(start_seconds)):
            raise ValueError("Session start must be a finite real number")
        self.start_seconds = float(start_seconds)


    def completed_epochs(self, heart: pd.DataFrame, motion: pd.DataFrame) -> pd.DataFrame:
        """Close intervals only through the earlier sensor's final observed timestamp.

        Args:
            heart (pd.DataFrame): Session-relative heart-rate observation times.
            motion (pd.DataFrame): Session-relative acceleration observation times.
        """
        ends = []
        for frame in (heart, motion):
            if "time" not in frame or frame.empty:
                raise ValueError("Both sensor streams require observation times")
            times = frame.time.to_numpy(dtype=float)
            if not np.isfinite(times).all():
                raise ValueError("Sensor times must be finite")
            ends.append(float(times.max()))
        count = int(np.floor((min(ends) - self.start_seconds) / 30.0))
        if count <= 0:
            raise ValueError("Sensor watermarks do not close a complete session epoch")
        return pd.DataFrame({"time": self.start_seconds + 30.0 * np.arange(count)})
