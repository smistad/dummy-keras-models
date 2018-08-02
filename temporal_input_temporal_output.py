"""
Simple model with one temporal image input and one temporal tensor output
"""

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input, Dense, Concatenate, Flatten, TimeDistributed, LSTM
from export import export_current_model

input = Input(shape=(None, 64, 64, 1))
x = TimeDistributed(Flatten())(input)
x = TimeDistributed(Dense(10))(x)
x = LSTM(10, return_sequences=True)(x)

model = Model(inputs=[input], outputs=[x])
print(model.summary())

output_name = model.output.name.split(':')[0]
print('Name of output node:', output_name)

export_current_model('models/temporal_input_temporal_output.pb', output_name)
