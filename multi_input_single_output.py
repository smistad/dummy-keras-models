"""
Simple model with two image inputs and one 1D tensor output
"""

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input, Dense, Concatenate, Flatten
from export import export_current_model

input1 = Input(shape=(64, 64, 1))
x1 = Flatten()(input1)

input2 = Input(shape=(64, 64, 1))
x2 = Flatten()(input2)

x = Concatenate()([x1, x2])
x = Dense(6)(x)

model = Model(inputs=[input1, input2], outputs=x)
print(model.summary())

output_name = model.output.name.split(':')[0]
print('Name of output node:', output_name)

export_current_model('models/multi_input_single_output.pb', output_name)
