from datetime import datetime
from keras import callbacks
from tensorflow import keras

from fizzbuzz.models import mlp
from os import path
import random
import numpy as np


def generate_dataset(starting_number=0, amount=0, randomrange=0, return_X_y=False):
    data = []
    target = []
    #print(randomrange)
    for _number in range(starting_number, amount):
        _number = random.randint(0, randomrange)
        _result = evaluate_keyword(_number)
        data.append(np.array(_number))
        target.append(np.array(_result))

    if return_X_y:
        print(data)
        print(target)
        return np.array(data), np.array(target)


def evaluate_keyword(number):
    if isinstance(number, int):
        if number % 3 == 0 and number % 5 == 0:
            return [0, 0, 0, 1]
        if number % 3 == 0:
            return [0, 0, 1, 0]
        elif number % 5 == 0:
            return [0, 1, 0, 0]
        else:
            return [1, 0, 0, 0]


features, labels = generate_dataset(return_X_y=True, amount=10000, randomrange=10000)
# print(features)


# normalize targets (so they are in range [0;1]
# labels /= labels.max()
# normalize features as well
# features /= features.max(axis=0)

# this is your task, implement the method get_model() in the models.py file
model = mlp.get_model()

run_name = 'fizzbuzz-{:%d-%b_%H-%M-%S}'.format(datetime.now())
dir_path = path.dirname(path.realpath(__file__))
log_dir = path.join(dir_path, 'logs', run_name)
print('logging to "{}"'.format(log_dir))
tb_callback = callbacks.TensorBoard(log_dir=log_dir)
model.fit(features, labels, batch_size=300, epochs=20000, validation_split=.3, callbacks=[tb_callback])
