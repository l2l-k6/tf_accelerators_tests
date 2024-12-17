#!/usr/bin/env python3

import tensorflow as tf


if __name__ == '__main__':
	print("Number of GPUs available: ", len(tf.config.list_physical_devices("GPU")))
