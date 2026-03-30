class Setup:
    def __init__(
        self,
        image: str,
        csv_name: str,
        pulse_per_sec: float,
        pulse_per_mv: float,
        sample_frequency: int,
        num_sampling_points: int
    ):
        self.image: str = image
        self.csv_name: str = csv_name
        self.pulse_per_sec: float = pulse_per_sec
        self.pulse_per_mv: float = pulse_per_mv
        self.sample_frequency: int = sample_frequency
        self.num_sampling_points: int = num_sampling_points

        self.hpulse: int = 0
        self.wpulse: int = 0
