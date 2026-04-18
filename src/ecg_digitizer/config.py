from dataclasses import dataclass


@dataclass
class DigitizerConfig:
    image: str
    csv_name: str
    pulse_per_sec: float
    sample_frequency: int
    num_sampling_points: int
