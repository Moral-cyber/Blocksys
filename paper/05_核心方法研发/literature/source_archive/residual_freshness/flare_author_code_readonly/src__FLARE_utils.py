import tensorflow as tf
import tensorflow_federated as tff
from models.model_creators_fns import *
import utils.config as config
import data_handler.data_fuctions as data
import os
from contextlib import contextmanager
import sys
import numpy as np

#OUTPUT SUPRESSION
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

NUM_CLIENTS = config.NUM_CLIENTS
lr = config.lr
MOMENTUM = config.MOMENTUM
p = config.p

def model_fn_for_clients(accumolator,server_weights,tau,u):
    keras_model = create_keras_model_for_FLARE(accumolator,server_weights,tau,u)           
    return tff.learning.from_keras_model(
                                keras_model,
                                input_spec=data.input_spec,
                                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])        

def model_fn():
  keras_model = create_keras_model()
  return tff.learning.from_keras_model(
                                keras_model,
                                input_spec=data.input_spec,
                                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

def acc_init():
  model = model_fn()
  client_weights = tff.learning.ModelWeights.from_model(model)
  accumolator =   tf.nest.map_structure(lambda x, y: x.assign(tf.zeros(tf.shape(y))),
                                        client_weights, client_weights)
  return [accumolator for i in range(config.NUM_CLIENTS)]

@tf.function
def sparsify_layer(layer, prun_percent):
  #current bug: canot excid 85%
  #input:   tff.learning.ModelWeights.from_model(model)
  #output:  tff.learning.ModelWeights.from_model(model)
    precent_to_zero = prun_percent # as %, its the precentege of the element to keep, if equal to 10, than only 10 largest will remain
    flat_layer =  tf.reshape(layer,[-1])
    k = tf.get_static_value(tf.size(flat_layer)) * (precent_to_zero/100)   # k is the number of eleement that is the top k
    b = tf.nn.top_k(tf.abs(flat_layer), tf.cast(tf.round(k)+2,tf.int32))
    kth = tf.reduce_min(b.values)
    #print(kth)
    mask = tf.greater(tf.abs(layer), kth * tf.ones_like(layer))
    prunned_layer = tf.multiply(layer, tf.cast(mask, tf.float32))
    return prunned_layer

@tf.function
def client_update_my_algo(models, dataset, server_weights, accumolator, client_optimizer, prun_percent, E):
  
  model_0 = models[0]
  model_1 = models[1]  
  # Initialize the client model with the current server weights.
  client_weights_new_0 = tff.learning.ModelWeights.from_model(model_0)
  client_weights_new_1 = tff.learning.ModelWeights.from_model(model_1)
  # Assign the server weights to the client model.
  tf.nest.map_structure(lambda x, y: x.assign(y),
                        client_weights_new_0, server_weights)
  steps_counter = 0
  for e in range(E):
    # first epoch
    for batch in dataset:
        if steps_counter < p:
          with tf.GradientTape() as tape:
            # Compute a forward pass on the batch of data
            outputs = model_0.forward_pass(batch)

          # Compute the corresponding gradient
          grads = tape.gradient(outputs.loss, client_weights_new_0.trainable)
          grads_and_vars = zip(grads, client_weights_new_0.trainable)

          # Apply the gradient using a client optimizer.
          client_optimizer.apply_gradients(grads_and_vars)
          # Assign the server weights to the client model.
          tf.nest.map_structure(lambda x, y: x.assign(y),
                                client_weights_new_1, client_weights_new_0)  
          steps_counter +=1
        else:
          with tf.GradientTape() as tape:
            # Compute a forward pass on the batch of data
            outputs = model_1.forward_pass(batch)

          # Compute the corresponding gradient
          grads = tape.gradient(outputs.loss, client_weights_new_1.trainable)
          grads_and_vars = zip(grads, client_weights_new_1.trainable)

          # Apply the gradient using a client optimizer.
          client_optimizer.apply_gradients(grads_and_vars)
          steps_counter +=1  
  
  #substructe new and old weights
  diference_client_weights = tf.nest.map_structure(lambda x, y: tf.subtract(x,y),
                                                    client_weights_new_1, server_weights)

  #add accumolator to the diference_client_weights
  diff_plus_acc = tf.nest.map_structure(lambda x, y: tf.add(x,y),
                                        diference_client_weights, accumolator)
  
  #create pruned weights diference
  pruned_client_diference_weights = tf.nest.map_structure(lambda x: sparsify_layer(x, prun_percent), 
                                                          diff_plus_acc)
  
  #create inverse pruned weights diference (by substructe the pruned from the not pruned)
  inverse_pruned_client_diference_weights = tf.nest.map_structure(lambda x, y: tf.subtract(x,y),
                                                                diff_plus_acc, pruned_client_diference_weights)
  
  #assign the  inverse pruned weights diference to the accumolator
  accumolator = inverse_pruned_client_diference_weights
  
  return pruned_client_diference_weights, accumolator

@tf.function
def client_update(model, dataset, server_weights, accumolator, client_optimizer, prun_percent, E):
  # Initialize the client model with the current server weights.
  client_weights = tff.learning.ModelWeights.from_model(model)

  # Assign the server weights to the client model.
  tf.nest.map_structure(lambda x, y: x.assign(y),
                        client_weights, server_weights)
  
  # Use the client_optimizer to update the local model.
  for e in range(int(E)):
    for batch in dataset:
      with tf.GradientTape() as tape:
        # Compute a forward pass on the batch of data
        outputs = model.forward_pass(batch)

      # Compute the corresponding gradient
      grads = tape.gradient(outputs.loss, client_weights.trainable)
      grads_and_vars = zip(grads, client_weights.trainable)

      # Apply the gradient using a client optimizer.
      client_optimizer.apply_gradients(grads_and_vars)

  #substructe new and old weights
  diference_client_weights = tf.nest.map_structure(lambda x, y: tf.subtract(x,y),
                                                    client_weights, server_weights)

  #add accumolator to the diference_client_weights
  diff_plus_acc = tf.nest.map_structure(lambda x, y: tf.add(x,y),
                                        diference_client_weights, accumolator)
  
  #create pruned weights diference
  pruned_client_diference_weights = tf.nest.map_structure(lambda x: sparsify_layer(x, prun_percent), 
                                                          diff_plus_acc)
  
  #create inverse pruned weights diference (by substructe the pruned from the not pruned)
  inverse_pruned_client_diference_weights = tf.nest.map_structure(lambda x, y: tf.subtract(x,y),
                                                                diff_plus_acc, pruned_client_diference_weights)
  
  #assign the  inverse pruned weights diference to the accumolator
  accumolator = inverse_pruned_client_diference_weights
  
  return pruned_client_diference_weights, accumolator

@tf.function
def FedAvg_client_update(model, dataset, server_weights, client_optimizer, E):
  # Initialize the client model with the current server weights.
  client_weights = tff.learning.ModelWeights.from_model(model)
  # Assign the server weights to the client model.
  tf.nest.map_structure(lambda x, y: x.assign(y),
                        client_weights, server_weights)

  # Use the client_optimizer to update the local model.
  for e in range(int(E)):
    for batch in dataset:
      with tf.GradientTape() as tape:
        # Compute a forward pass on the batch of data
        outputs = model.forward_pass(batch)

      # Compute the corresponding gradient
      grads = tape.gradient(outputs.loss, client_weights.trainable)
      grads_and_vars = zip(grads, client_weights.trainable)

      # Apply the gradient using a client optimizer.
      client_optimizer.apply_gradients(grads_and_vars)
  return client_weights

@tff.tf_computation
def server_init():
  model = model_fn()
  return tff.learning.ModelWeights.from_model(model)

@tff.federated_computation
def initialize_fn():
  return tff.federated_value(server_init(), tff.SERVER)

@tf.function
def server_update(model, mean_client_diference, server_weights):
  return tf.nest.map_structure(lambda x, y: tf.add(x,y),
                                    mean_client_diference, server_weights)

#types defenition
whimsy_model = model_fn()
tf_dataset_type = tff.SequenceType(whimsy_model.input_spec)
#print(str(tf_dataset_type))
model_weights_type = server_init.type_signature.result
#print(str(model_weights_type))
prun_percent_type = tf.constant(1, dtype = tf.float32).dtype
#print(str(prun_percent_type))

#federated types
federated_server_type = tff.FederatedType(model_weights_type, tff.SERVER)
federated_dataset_type = tff.FederatedType(tf_dataset_type, tff.CLIENTS)
federated_clients_type = tff.FederatedType(model_weights_type, tff.CLIENTS)
federated_prun_percent_type = tff.FederatedType(prun_percent_type, tff.CLIENTS)

@tff.tf_computation(model_weights_type, model_weights_type)
def server_update_fn(weights_difference_mean, server_weights):
  model = model_fn()
  return server_update(model, weights_difference_mean ,server_weights)

@tff.federated_computation(federated_server_type,federated_server_type)
def server_update_fn(weights_difference_mean ,server_weights):
  return tff.federated_map(server_update_fn, (weights_difference_mean ,server_weights))

@tff.tf_computation(tf_dataset_type, model_weights_type, model_weights_type, prun_percent_type, prun_percent_type, prun_percent_type, prun_percent_type,prun_percent_type)
def client_update_fn(tf_dataset, server_weights, accumolator, prun_percent, learning_rate, E, tau,u):
  models_0 = model_fn_for_clients(accumolator,server_weights,tau,u)
  models_1 = model_fn()
  models = [models_0,models_1]
  client_optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=MOMENTUM)
  pruned_client_weights, accumolator = client_update_my_algo(models, tf_dataset, server_weights, accumolator, client_optimizer, prun_percent, E) 
  return pruned_client_weights, accumolator
 
@tff.federated_computation(federated_dataset_type, tff.type_at_clients(model_weights_type), federated_clients_type, federated_prun_percent_type, federated_prun_percent_type, federated_prun_percent_type,federated_prun_percent_type,federated_prun_percent_type)
def client_update_fn(tf_dataset, server_weights, accumolator, prun_percent, learning_rate, E, tau,u):
  return tff.federated_map(client_update_fn, (tf_dataset, server_weights, accumolator, prun_percent, learning_rate, E, tau,u))

@tff.tf_computation(tf_dataset_type, model_weights_type, model_weights_type, prun_percent_type, prun_percent_type, prun_percent_type)
def Second_algo_client_update_fn(tf_dataset, server_weights, accumolator, prun_percent, learning_rate, E):
  #model = model_fn_for_clients(accumolator,server_weights,tau,u)  
  model = model_fn() #build regular model
  client_optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=MOMENTUM)
  pruned_client_weights, accumolator = client_update(model, tf_dataset, server_weights, accumolator, client_optimizer, prun_percent, E) 
  return pruned_client_weights, accumolator
 
@tff.federated_computation(federated_dataset_type, tff.type_at_clients(model_weights_type), federated_clients_type, federated_prun_percent_type, federated_prun_percent_type, federated_prun_percent_type)
def Second_algo_client_update_fn(tf_dataset, server_weights, accumolator, prun_percent, learning_rate, E):
  return tff.federated_map(Second_algo_client_update_fn, (tf_dataset, server_weights, accumolator, prun_percent, learning_rate, E))

@tff.tf_computation(tf_dataset_type, model_weights_type, prun_percent_type)
def FedAvg_client_update_fn(tf_dataset, server_weights, E):
  model = model_fn()
  client_optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=MOMENTUM)
  client_weights = FedAvg_client_update(model, tf_dataset, server_weights, client_optimizer, E)
  return client_weights

@tff.federated_computation(federated_dataset_type, tff.type_at_clients(model_weights_type), federated_prun_percent_type)
def FedAvg_client_update_fn(tf_dataset, server_weights, E):
  return tff.federated_map(FedAvg_client_update_fn, (tf_dataset, server_weights, E))

@tf.function
def FedAvg_server_update(model, mean_client_weights):
  return mean_client_weights

@tff.tf_computation(model_weights_type)
def FedAvg_server_updatefn(mean_client_weights):
  model = model_fn()
  return FedAvg_server_update(model, mean_client_weights)

@tff.federated_computation(federated_server_type)
def FedAvg_server_update_fn(server_weights):
  return tff.federated_map(FedAvg_server_updatefn, server_weights)

@tff.federated_computation(federated_server_type, federated_clients_type, federated_dataset_type, federated_prun_percent_type, federated_prun_percent_type, federated_prun_percent_type, federated_prun_percent_type,federated_prun_percent_type)
def next_fn(server_weights, accumoltors, federated_dataset, prun_percent, learning_rate, E, tau,u):
  # Broadcast the server weights to the clients. server -> clients
  server_weights_at_client = tff.federated_broadcast(server_weights)

  # Each client computes their updated weights. clients -> clients
  pruned_client_weights_diference, accumoltors = client_update_fn(federated_dataset, server_weights_at_client, accumoltors, prun_percent, learning_rate, E, tau,u)

  # The server averages these weights_difference. clients -> server
  weights_difference_mean = tff.federated_mean(pruned_client_weights_diference)

  #the server adds the old weights to weights_difference and updates its model. server -> server
  server_weights = server_update_fn(weights_difference_mean ,server_weights)

  return server_weights, accumoltors

@tff.federated_computation(federated_server_type, federated_clients_type, federated_dataset_type, federated_prun_percent_type, federated_prun_percent_type, federated_prun_percent_type)
def Second_algo_next_fn(server_weights, accumoltors, federated_dataset, prun_percent, learning_rate, E):
  # Broadcast the server weights to the clients. server -> clients
  server_weights_at_client = tff.federated_broadcast(server_weights)

  # Each client computes their updated weights. clients -> clients
  pruned_client_weights_diference, accumoltors = Second_algo_client_update_fn(federated_dataset, server_weights_at_client, accumoltors, prun_percent, learning_rate,E)

  # The server averages these weights_difference. clients -> server
  weights_difference_mean = tff.federated_mean(pruned_client_weights_diference)

  #the server adds the old weights to weights_difference and updates its model. server -> server
  server_weights = server_update_fn(weights_difference_mean ,server_weights)

  return server_weights, accumoltors

@tff.federated_computation(federated_server_type, federated_dataset_type,federated_prun_percent_type)
def FedAvg_next_fn(server_weights, federated_dataset, E):
  # Broadcast the server weights to the clients.
  server_weights_at_client = tff.federated_broadcast(server_weights)

  # Each client computes their updated weights.
  client_weights = FedAvg_client_update_fn(federated_dataset, server_weights_at_client, E)

  # The server averages these updates.
  mean_client_weights = tff.federated_mean(client_weights)

  # The server updates its model.
  server_weights = FedAvg_server_update_fn(mean_client_weights)

  return server_weights

def calc_multypliers_FFL(history_federeted,second_algo_server_state,PRUN_PERCENT,E,central_emnist_test):       
        with suppress_stdout():
                eval = evaluate(second_algo_server_state,central_emnist_test)
        current_loss = eval[0]
        First_loss = np.array(history_federeted)[:,0][0]
        prun_percent_FFL = PRUN_PERCENT * np.power((First_loss/current_loss),1/3)
        E_FFL = np.round(E * np.power((current_loss/First_loss),1/3))
        if E_FFL < 1:
                E_FFL = 1  
                               
        return prun_percent_FFL, E_FFL

def evaluate(server_state,central_emnist_test):
  keras_model = create_keras_model()
  keras_model.compile(
      #loss=Loss_Fn(keras_model.losses),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]  
  )
  server_state.assign_weights_to(keras_model)
  res = keras_model.evaluate(central_emnist_test)
  return res