"""
Simple model with one image input and two 1D tensor outputs
"""

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input, Dense, Concatenate, Flatten
from export import export_current_model

input1 = Input(shape=(64, 64, 1))
x = Flatten()(input1)

x = Dense(10)(x)
x1 = Dense(6)(x)
x2 = Dense(6)(x)

model = Model(inputs=[input1], outputs=[x1, x2])
print(model.summary())

output_names = []
for output in model.output:
    output_name = output.name.split(':')[0]
    print('Name of output node:', output_name)
    output_names.append(output_name)

export_current_model('models/single_input_multi_output.pb', output_names)
