class Setup:
    def __init__(
        self,
        image,                  
        csv_name,              
        pulse_per_sec,          
        pulse_per_mv,          
        sample_frequency,       
        num_sampling_points     
    ):
        self.image = image
        self.csv_name = csv_name
        self.pulse_per_sec = pulse_per_sec
        self.pulse_per_mv = pulse_per_mv
        self.sample_frequency = sample_frequency
        self.num_sampling_points = num_sampling_points

        self.hpulse = 0
        self.wpulse = 0