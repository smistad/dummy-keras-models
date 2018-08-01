import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.tools.freeze_graph import freeze_graph


def export_current_model(filename, output_nodes):
    K.set_learning_phase(0) # Turn off learning phase

    # Save graph as well
    print('Writing checkpoint to disk..')
    tf.train.Saver().save(K.get_session(), '/tmp/model.ckpt') # Save weights
    input_graph_name = '/tmp/tmp_graph.pb'
    print('Writing graph to disk..')
    tf.train.write_graph(K.get_session().graph.as_graph_def(), logdir='', name=input_graph_name) # Save graph

    # Combine the two
    print('Combining graph and weights..')
    if type(output_nodes) is list:
        output = ''
        for x in output_nodes:
            output += x + ','
        output = output[:len(output)-1] # Remove last ,
    else:
        output = output_nodes
    freeze_graph(input_graph_name, '', False, '/tmp/model.ckpt', output, 'save/restore_all', 'save/Const:0', filename, False, '')
