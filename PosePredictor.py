from tensorflow import nn

class PosePredictor:
    def __init__(self, pretrained_model = []):
        self.model = pretrained_model

        self.__first_exe = True
        self.__counter_exe = 0

    def update_data(self, data, label):
        if not self.__first_exe:
            self.__counter_exe += 1
        else:
            self.__first_exe = False
            self.__counter_exe = 0