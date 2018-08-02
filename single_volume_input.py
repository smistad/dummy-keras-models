"""
Simple model with one 3D image input and one tensor output
"""

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input, Dense, Concatenate, Flatten
from export import export_current_model

input = Input(shape=(64, 64, 64, 1))
x = Flatten()(input)

x = Dense(10)(x)

model = Model(inputs=[input], outputs=[x])
print(model.summary())

output_name = model.output.name.split(':')[0]
print('Name of output node:', output_name)

export_current_model('models/single_volume_input.pb', output_name)
