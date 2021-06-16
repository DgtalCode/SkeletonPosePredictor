from tensorflow import nn
import time
import json

class PosePredictor:
    def __init__(self, pretrained_model={}):
        self.__model = pretrained_model
        self.__model_data = []
        self.__model_label = 0

        self.__first_exe = True
        self.__counter_exe = 0

        self.__time_exe = 0

        self.__task = False

    def update_data(self, data):
        if self.__task:
            if time.time() % 1 < 0.1 and time.time() - self.__time_exe < 10:
                print("WRITE WILL START IN {} SECONDS".format(10 - round(time.time() - self.__time_exe)))

        if not self.__first_exe:
            if time.time() - self.__time_exe > 10:
                self.__model_data.append(data)
                self.__counter_exe += 1
                if self.__counter_exe > 40:
                    self.__model.update({self.__model_label: self.__model_data})
                    self.__task = False
                    self.__counter_exe = 0
                    self.__first_exe = True
                    print("WRITE COMPLETED")
        else:
            if self.__task:
                print("WRITE WILL START IN {} SECONDS".format(10))
                self.__first_exe = False
                self.__counter_exe = 0
                self.__time_exe = time.time()

    def start_writing(self, label):
        self.__task = True
        self.__model_label = label

    def save_data(self):
        with open('data.json', 'w') as writefile:
            json.dump(self.__model, writefile)
