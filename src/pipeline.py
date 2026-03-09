class Pipeline:
    """
    Generic pipeline executor.

    Each stage must be a callable that receives and returns a dictionary
    containing the pipeline state.
    """

    def __init__(self, stages, verbose=False):

        if not isinstance(stages, list):
            raise TypeError("Stages must be provided as a list.")

        self.stages = stages
        self.verbose = verbose

    def run(self, data):
        """
        Execute all pipeline stages sequentially.
        """

        if not isinstance(data, dict):
            raise TypeError("Pipeline input must be a dictionary.")

        for stage in self.stages:

            stage_name = stage.__name__

            if self.verbose:
                print(f"PIPELINE: running stage -> {stage_name}")

            try:
                data = stage(data)

            except Exception as e:

                raise RuntimeError(
                    f"Pipeline failed at stage '{stage_name}'"
                ) from e

        return data