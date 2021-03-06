from keras import layers, engine


def get_model():
    # we need a layer that acts as input.
    # shape of that input has to be known and depends on data.
    input_layer = layers.Input(shape=(1,))

    # hidden layers are the model's power to fit data.
    # number of neurons and type of layers are crucial.
    # idea behind decreasing number of units per layer:
    # increase the "abstraction" in each layer...


    # last layer represents output.
    # activation of each neuron corresponds to the models decision of
    # choosing that class.
    # softmax ensures that all activations summed up are equal to 1.
    # this lets one interpret that output as a probability
    hidden_layer = layers.Dense(units=100, activation='relu')(input_layer)
    #hidden_layer = layers.Dense(units=20, activation='relu')(hidden_layer)
    output_layer = layers.Dense(units=4, activation="relu")(hidden_layer)
    # actual creation of the model with in- and output layers
    model = engine.Model(inputs=[input_layer], outputs=[output_layer])
    # transform into a trainable model by specifying the optimizing function
    # (here stochastic gradient descent),
    # as well as the loss (eg. how big of an error is produced by the model)
    # track the model's accuracy as an additional metric (only possible for classification)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
