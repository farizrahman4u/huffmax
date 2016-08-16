from keras.layers import Layer, Dense, InputSpec, Lambda, Input
from keras import activations
from keras import backend as K
import numpy as np
import warnings
import sys
import datetime


sys.setrecursionlimit(10000000)



def arange(n):
	if K._BACKEND == 'theano':
		import theano.tensor as T
		return T.arange(n)
	elif K._BACKEND == 'tensorflow':
		import tensorflow as tf
		return tf.range(n)


class Huffmax(Layer):

	def __init__(self, nb_classes, frequency_table=None, mode=None, verbose=False, **kwargs):
		if frequency_table is None:
			frequency_table = [1] * nb_classes
		self.frequency_table = frequency_table
		if 'weights' in kwargs:
			self.initial_weights = kwargs['weights']
			del kwargs['weights']
		else:
			self.initial_weights = None
		self.mode = mode
		self.nb_classes = nb_classes
		self.kwargs = kwargs
		self.verbose = verbose
		super(Huffmax, self).__init__()

	def build(self, input_shape):
		if self.verbose:
			print 'build started'
		self.input_spec = [InputSpec(shape=input_shape[0]), InputSpec(shape=(input_shape[1]))]
		# Calculate number of nodes required from nb_classes
		log = np.floor(np.log(self.nb_classes) / np.log(2))
		nb_nodes = np.power(2, log + 1) - 1
		nb_nodes += self.nb_classes - np.power(2, log)

		def combine_nodes(left, right):
			# Creates a parent node, sets left and right as its children, and returns it.
			parent_node = Dense(output_dim=2, **self.kwargs)
			parent_node(Input(batch_shape=input_shape[0]))
			output_shape = parent_node.get_output_shape_for(input_shape[0])
			assert len(output_shape) == 2
			assert output_shape[1] == 2
			parent_node.left = left
			parent_node.right = right
			parent_node.code = left.code + right.code
			parent_node.frequency = left.frequency + right.frequency
			return parent_node
		# Generate leaves of Huffman tree.
		leaves = [Lambda(lambda x: K.cast(x * 0 + i, dtype='int32')) for i in range(self.nb_classes)]
		# Set attribs for leaves
		for l in range(len(leaves)):
			leaf = leaves[l]
			leaf.built = True
			leaf.code = [l]
			leaf.frequency = self.frequency_table[l]
		# Build Huffman tree.
		if self.verbose:
			print 'Building huffman tree...'
		un_merged_nodes = leaves[:]
		self.nodes = []
		frequencies = [l.frequency for l in leaves]
		# We keep merging 2 least frequency nodes, until only the root node remains. Classic Huffman tree, nothing fancy.
		prev_p = 0
		while len(un_merged_nodes) > 1:
			p = int(100. * float(nb_nodes - len(un_merged_nodes) - 1) / nb_nodes)
			if self.verbose:
				if p > prev_p:
					sys.stdout.write('\r' + str(p) + ' %')
					prev_p = p
			min_frequency_node = np.argmin(frequencies)
			left = un_merged_nodes.pop(min_frequency_node)
			frequencies.pop(min_frequency_node)
			min_frequency_node = np.argmin(frequencies)
			right = un_merged_nodes.pop(min_frequency_node)
			frequencies.pop(min_frequency_node)
			parent_node = combine_nodes(left, right)
			self.nodes += [parent_node]
			un_merged_nodes += [parent_node]
			frequencies += [parent_node.frequency]
		if self.verbose:
			print 'Huffman tree build complete'
		self.root_node = un_merged_nodes[0]
		self.nodes += [self.root_node]
		self.node_indices = {self.nodes[i]: i for i in range(len(self.nodes))}
		self.leaves = leaves
		# Set parameters and weights
		self.trainable_weights = []
		self.regularizers = []
		self.constraints = {}
		for node in self.nodes:
			if self.initial_weights:
				nb_weights = len(node.get_weights())
				node.set_weights(self.initial_weights[:nb_weights])
				self.initial_weights = self.initial_weights[nb_weights:]
			self.trainable_weights += node.trainable_weights
			self.regularizers += node.regularizers
			for key in node.constraints.keys():
				self.constraints[key] = node.constraints[key]
		del self.initial_weights
		# Set paths and huffman codes
		self.paths = []
		self.huffman_codes = []
		self.one_hot_huffman_codes = []
		for i in range(self.nb_classes):
			path, huffman_code = self._traverse_huffman_tree(i)
			self.paths += [path]
			self.huffman_codes += [huffman_code]
			one_hot_huffman_code = [([1, 0] if c == 0 else [0, 1]) for c in huffman_code]
			self.one_hot_huffman_codes += [one_hot_huffman_code]
		self.max_tree_depth = max(map(len, self.huffman_codes))
		for huffman_code in self.huffman_codes:
			huffman_code += [0] * (self.max_tree_depth - len(huffman_code))
		self.padded_one_hot_huffman_codes = self.one_hot_huffman_codes[:]
		for one_hot_huffman_code in self.padded_one_hot_huffman_codes:
			one_hot_huffman_code += [[1, 1]] * (self.max_tree_depth - len(one_hot_huffman_code))
		input_dim = self.input_spec[0].shape[1]

		if self.verbose:
			print 'Generating matrices...'
		# Weights
		self.W = K.variable(np.array([node.W.get_value() for node in self.nodes]))
		self.b = K.variable(np.array([node.b.get_value() for node in self.nodes])) if self.root_node.bias else K.zeros((len(self.nodes), 2))
		self.trainable_weights = [self.W]
		if False and self.root_node.bias:
			self.trainable_weights += [self.b]
		# Class -> path map
		self.class_path_map = K.variable(np.array([[self.node_indices[node] for node in path + [self.root_node] * (self.max_tree_depth - len(path))] for path in self.paths]), dtype='int32')
		super(Huffmax, self).build(input_shape)
		if self.verbose:
			print 'Matrices generation complete.'

	def _traverse_huffman_tree(self, leaf_index):
		# Finds the path and huffman code for a given leaf in the huffman tree. 0 is left, 1 is right.
		leaf = self.leaves[leaf_index]
		current_node = self.root_node
		huffman_code = []
		path = []
		while current_node != leaf:
			path += [current_node]
			if leaf_index in current_node.left.code:
				huffman_code += [0]
				current_node = current_node.left
			else:
				current_node = current_node.right
				huffman_code += [1]
		return path, huffman_code

	def call(self, x, mask=None):
		input_vector = x[0]  # batch_size, input_dim
		target_classes = x[1]  # batch_size, nb_req_classes
		# Make sure target_classes is of integer type
		if K.dtype(target_classes) != 'int32':
			target_classes = K.cast(target_classes, 'int32')
		input_dim = self.input_spec[0].shape[1]
		nb_req_classes = self.input_spec[1].shape[1]
		big_W = self.W  # nb_nodes, input_dim, 2
		big_b = self.b  # nb_nodes, 2
		path_lengths = map(len, self.paths)
		# Tensorify huffman codes
		huffman_codes = K.variable(np.array(self.padded_one_hot_huffman_codes))  # nb_classes, max_tree_depth, 2
		req_nodes = K.gather(self.class_path_map, target_classes)  # batch_size, nb_req_classes, max_tree_depth
		req_W = K.gather(big_W, req_nodes)  # batch_size, nb_req_classes, max_tree_depth, input_dim, 2
		#req_b = K.gather(big_b, req_nodes)  # batch_size, nb_req_classes, max_tree_depth, 2
		y = K.batch_dot(input_vector, req_W, axes=(1, 3))# + req_b  # batch_size, nb_req_classes, max_tree_depth, 2
		y = K.exp(y - K.max(y))
		y /= K.sum(y, axis=-1, keepdims=True)
		req_huffman_codes = K.gather(huffman_codes, target_classes)  # batch_size, nb_req_classes, max_tree_depth, 2
		# Tree traversal
		return K.prod(K.sum(y * req_huffman_codes, axis=-1), axis=-1)

	def get_output_shape_for(self, input_shape):
		return (input_shape[0][0], input_shape[1][1])

	def get_config(self):
		config = {'nb_classes': self.nb_classes,
				  'mode': self.mode,
				  'frequency_table': self.frequency_table,
				  'kwargs': self.kwargs
				  }
		base_config = super(Huffmax, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
