from keras.layers import Layer, Dense
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

	def __init__(self, nb_classes, activation='softmax', frequency_table=None, mode=None, verbose=False, **kwargs):
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
		self.activation = activations.get(activation)
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
		leaves = [Lambda(lambda x: x) for _ in range(self.nb_classes)]
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
		big_W = []
		big_b = []
		if self.verbose:
			print 'Generating matrices...'
		prev_p = 0
		big_W_mask = []
		big_b_mask = []
		for i in range(len(self.paths)):
			p = int(100. * (i + 0.) / len(self.paths))
			if self.verbose:
				if p > prev_p:
					sys.stdout.write('\r' + str(p) + ' %')
					prev_p = p
			path = self.paths[i]
			W = [node.W.get_value() for node in path]
			b = [node.b.get_value() for node in path]
			W = np.transpose(np.array(W), (1, 0, 2))
			W = np.reshape(W, (input_dim, len(path) * 2))
			b = np.array(b)
			b = np.reshape(b, (len(path) * 2))
			# We need masks, since different classes have huffman codes of different lengths
			W_mask = np.ones(W.shape)
			b_mask = np.ones(b.shape)
			if len(path) < self.max_tree_depth:
				diff = self.max_tree_depth - len(path)
				W = np.concatenate([W, np.zeros((input_dim, diff * 2))], axis=1)
				W_mask = np.concatenate([W_mask, np.zeros((input_dim, diff * 2))], axis=1)
				b = np.concatenate([b, np.zeros((diff * 2, ))])
				b_mask = np.concatenate([b_mask, np.zeros((diff * 2, ))])
			big_W += [W]
			big_b += [b]
			big_W_mask += [W_mask]
			big_b_mask += [b_mask]
		big_W, big_b, big_W_mask, big_b_mask = map(lambda x: K.variable(np.array(x)), [big_W, big_b, big_W_mask, big_b_mask])
		self.W = big_W
		self.b = big_b
		self.W_mask = big_W_mask
		self.b_mask = big_b_mask
		self.trainable_weights = [self.W]
		if self.root_node.bias:
			self.trainable_weights += [self.b]
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
		big_W = self.W  # nb_classes, input_dim, max_tree_depth * 2
		big_b = self.b  # nb_classes, max_tree_depth * 2
		path_lengths = map(len, self.paths)
		# Tensorify huffman codes
		huffman_codes = K.variable(np.array(self.padded_one_hot_huffman_codes))  # nb_classes, max_tree_depth, 2
		if self.mode == None:
			if len(self.frequency_table) == 1:
				mode = 0
			else:
				mode = 1
		else:
			mode = self.mode
		if mode == 0:
			# In this mode, we pick required regions of the weights and dot them with the input
			# Apply masks so that gradients do not flow into the padded regions of the weights
			big_W *= self.W_mask  # nb_classes, input_dim, max_tree_depth * 2
			big_b *= self.b_mask  # nb_classes, max_tree_depth * 2
			req_W = K.gather(big_W, target_classes)  # batch_size, nb_req_classes, input_dim, max_tree_depth * 2
			req_b = K.gather(big_b, target_classes)  # batch_size, nb_req_classes, max_tree_depth * 2
			# Compute dot
			outputs_at_nodes = K.batch_dot(input_vector, req_W, axes=(1, 2)) + req_b  # batch_size, nb_req_classes, max_tree_depth * 2
			# Apply activation function
			outputs_at_nodes = K.reshape(outputs_at_nodes, (-1, 2))  # batch_size * nb_req_classes * max_tree_depth, 2
			p = self.activation(outputs_at_nodes)  # batch_size * nb_req_classes * max_tree_depth, 2
			p = K.reshape(p, (-1, nb_req_classes, self.max_tree_depth, 2))  # batch_size, nb_req_classes, max_tree_depth, 2
			# Pick required huffman codes
			req_huffman_codes = K.gather(huffman_codes, target_classes)  # batch_size, nb_req_classes, max_tree_depth, 2
			# Tree traversal
			p = K.prod(K.sum(p * req_huffman_codes, axis=-1), axis=-1)  # batch_size, nb_req_classes
			return p			
		elif self.mode == 1:
			# In this mode, we compute the node outputs for each class, and then pick the required outputs
			outputs = []
			for i in range(self.nb_classes):
				path_length = path_lengths[i]
				# Get required portions of the weights
				W = big_W[i, :, :path_length * 2]  # input_dim, nb_nodes * 2
				b = big_b[i, :path_length * 2]  # nb_nodes * 2
				# Compute dot
				output = K.dot(input_vector, W) + b  # batch_size, nb_nodes * 2
				# Apply activation function
				output = K.reshape(output, (-1, 2))  # batch_size, nb_nodes, 2
				output = self.activation(output)  # batch_size * nb_nodes, 2
				output = K.reshape(output, (-1, path_length, 2))  # batch_size, nb_nodes, 2
				# Normalize lengths
				if path_length < self.max_tree_depth:
					output = K.concatenate([output] + [output[:, :1, :] * 0 + 0.5] * (self.max_tree_depth - path_length), axis=1)  # batch_size, max_tree_depth, 2
				outputs += [output]
			outputs = K.pack(outputs)  # nb_classes, batch_size, max_tree_depth, 2
			# Pick required outputs
			req_outputs = K.gather(outputs, target_classes)  # batch_size, nb_req_classes, batch_size, max_tree_depth, 2
			# Convert to square tensor
			req_outputs = K.permute_dimensions(req_outputs, (0, 2, 1, 3, 4))  # batch_size, batch_size, nb_req_classes, max_tree_depth, 2
			# Pick diagonal elements
			req_outputs = K.reshape(req_outputs, (-1, nb_req_classes, self.max_tree_depth, 2))  # batch_size * batch_size, nb_req_classes, max_tree_depth, 2
			batch_size = K.shape(input_vector)[0]
			diag_indices = arange(batch_size) * (batch_size + 1)  # batch_size
			diag_elems = K.gather(req_outputs, diag_indices)  # batch_size, nb_req_classes, max_tree_depth, 2
			# Gather required huffman codes
			req_huffman_codes = K.gather(huffman_codes, target_classes)  # nb_req_classes, max_tree_depth, 2
			# Tree traversal
			req_probs = K.prod(K.sum(diag_elems * req_huffman_codes, axis=-1), axis=-1)  # batch_size, nb_req_classes
			return req_probs

	def get_output_shape_for(self, input_shape):
		return (input_shape[0][0], input_shape[1][1])

	def get_config(self):
		config = {'nb_classes': self.nb_classes,
				  'mode': self.mode,
				  'activation': self.activation.__name__,
				  'frequency_table': self.frequency_table,
				  'kwargs': self.kwargs
				  }
		base_config = super(Huffmax, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
