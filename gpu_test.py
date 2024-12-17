#!/usr/bin/env python3

print("Interpeter loaded successfully.")

import tensorflow as tf
print("Number of GPUs available: ", len(tf.config.list_physical_devices("GPU")))
