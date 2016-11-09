__all__ = []

class NNLayer():
    """NNLayer class

    Base class for Neural Network layers
    """
    def get_outputs(self):
        """
        Function for generating outputs considering the entire network
        structure for a layer.
        """
        return NotImplementedError

    def reset_state(self):
        """
        Function for resetting the state parameters in case the network
        is to be reset to initial network states.
        """
        return NotImplementedError
