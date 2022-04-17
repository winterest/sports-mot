"""Logger
"""
import os


class Logging(object):
    def __init__(self, time_str):
        self.time_stamp = time_str
        # Path
        path = os.path.join("./checkpoints", self.time_stamp)
        # Create the directory
        os.makedirs(path)
        print("Directory {} created".format(path))

        with open("./logs/training_{}.log".format(time_str), "a") as file:
            file.write("===starting training===\n")

    def write(self, context):
        with open(
            "./logs/training_{}.log".format(self.time_stamp), "a"
        ) as file:
            file.write(context)
