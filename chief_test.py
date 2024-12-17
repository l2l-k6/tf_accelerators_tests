#!/usr/bin/env python3

import os
import json

os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "worker": ["chief:port", "worker1:port"],
    },
   "task": {"type": "worker", "index": 0}
})

import tensorflow as tf
strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope() as scope:
	model = tf.keras.Sequential(
		[tf.keras.layers.Dense(1, input_shape=(1,),
		kernel_regularizer=tf.keras.regularizers.L2(1e-4))])
	model.compile(loss='mse', optimizer='sgd')

dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(100).batch(10)

if __name__ == "__main__":
	print(f'We are supposed to be the chief worker, here is our TF_CONFIG: {os.environ["TF_CONFIG"]}')

	model.fit(dataset, epochs=2)
	model.evaluate(dataset)
