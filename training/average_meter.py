class AverageMeter:
    """Class for maintaining average, sum and count of metrics.
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Starts counting metrics from scratch.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Predicts a segmentation mask for a specific image
        Parameters
        ----------
        val : float
            Received value.
        n : int
            Number of times the value has been received.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
