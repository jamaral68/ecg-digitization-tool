class Setup:
    def __init__(
        self,
        image,
        strategy,
        thres_value,
        dilation,
        perc_space_leads,
        layout,
        perc_max_dist,
        pulse,
        rhythm,
        verbose,
        mmpsec,
        mmpmv,
        pulse_width_mm,
        pulse_height_mm,
        pulse_per_sec,
        pulse_per_mv,
        sample_frequency,
        time_lead,
        location,
        num_sampling_points,
        lower,
        upper,
        kSize2d,
        kSize1d,
        template,     
        csv_name      
    ):
        self.image = image
        self.strategy = strategy
        self.thres_value = thres_value
        self.dilation = dilation
        self.perc_space_leads = perc_space_leads
        self.layout = layout
        self.perc_max_dist = perc_max_dist
        self.pulse = pulse
        self.rhythm = rhythm
        self.pulse_width_mm = pulse_width_mm
        self.pulse_height_mm = pulse_height_mm
        self.pulse_per_sec = pulse_per_sec
        self.pulse_per_mv = pulse_per_mv
        self.sample_frequency = sample_frequency
        self.time_lead = time_lead
        self.location = location
        self.num_sampling_points = num_sampling_points
        self.verbose = verbose
        self.lower = lower
        self.upper = upper
        self.kSize2d = kSize2d
        self.kSize1d = kSize1d
        self.mmpsec = mmpsec
        self.mmpmv = mmpmv
        self.template = template
        self.csv_name = csv_name
        self.hpulse = 0
        self.wpulse = 0

        # Automatically set leads based on layout
        self.lt_leads = self._set_leads()

    def _set_leads(self):
        """
        Define lead names based on the layout (rows x columns)
        """
        rows, cols = self.layout

        if cols == 4 and rows == 3:
            return [
                ['I', 'aVR', 'V1', 'V4'],
                ['II', 'aVL', 'V2', 'V5'],
                ['III', 'aVF', 'V3', 'V6'],
                ['II']
            ]
        elif cols == 2:
            raise NotImplementedError('Layout with 2 columns not implemented')
        elif cols == 1:
            return [
                ['I'],
                ['II'],
                ['III'],
                ['aVR'],
                ['aVL'],
                ['aVF'],
                ['V1'],
                ['V2'],
                ['V3'],
                ['V4'],
                ['V5'],
                ['V6']
            ]
        else:
            raise ValueError('Columns must be 4, 2, or 1')