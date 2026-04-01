class Setup:
    def __init__(
        self,
        image: str,
        csv_name: str,
        pulse_per_sec: float,
        sample_frequency: int,
        num_sampling_points: int
    ):
        self.image = image
        self.csv_name = csv_name
        self.pulse_per_sec = pulse_per_sec
        self.sample_frequency = sample_frequency
        self.num_sampling_points = num_sampling_points