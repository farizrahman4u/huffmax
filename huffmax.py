from keras.layers import Layer, Dense, InputSpec, Lambda, Input
from keras import activations
from keras import backend as K
from keras import initializations
from keras import regularizers
from keras import constraints
import numpy as np
import warnings
import sys
import datetime


sys.setrecursionlimit(10000000)



def arange(n, step=1):
	if K._BACKEND == 'theano':
		import theano.tensor as T
		return T.arange(0, n, step)
	elif K._BACKEND == 'tensorflow':
		import tensorflow as tf
		return tf.range(0, n, step)

def zeros(n):
	if K._BACKEND == 'theano':
		import theano.tensor as T
		return T.zeros(n)
	elif K._BACKEND == 'tensorflow':
		import tensorflow as tf
		return tf.zeros(n)

class Node(object):
	pass

class Huffmax(Layer):
	'''
	inputs : [2D vector; float (batch_size, input_dim), 2D target classes; int (batch_size, nb_required_classes)]
	output: [2D probabilities; float (batch_size, nb_required_classes)]
	'''
	def __init__(self, nb_classes, frequency_table=None, mode=0, init='glorot_uniform', weights=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None,
				 W_constraint=None, b_constraint=None,
				 bias=True, verbose=False, **kwargs):
		'''
		# Arguments:
		nb_classes: Number of classes.
		frequency_table: list. Frequency of each class. More frequent classes will have shorter huffman codes.
		mode: integer. One of [0, 1]
		verbose: boolean. Set to true to see the progress of building huffman tree. 
		'''
		self.nb_classes = nb_classes
		if frequency_table is None:
			frequency_table = [1] * nb_classes
		self.frequency_table = frequency_table
		self.mode = mode
		self.init = initializations.get(init)
		self.W_regularizer = regularizers.get(W_regularizer)
		self.b_regularizer = regularizers.get(b_regularizer)
		self.activity_regularizer = regularizers.get(activity_regularizer)
		self.W_constraint = constraints.get(W_constraint)
		self.b_constraint = constraints.get(b_constraint)
		self.bias = bias
		self.initial_weights = weights
		self.verbose = verbose
		super(Huffmax, self).__init__(**kwargs)

	def build(self, input_shape):
		if self.verbose:
			print('Build started')
		if type(input_shape) == list:
			self.input_spec = [InputSpec(shape=input_shape[0]), InputSpec(shape=(input_shape[1]))]
		else:
			self.input_spec = [InputSpec(shape=input_shape)]
			input_shape = [input_shape, None]
		input_dim = input_shape[0][1]
		def combine_nodes(left, right):
			parent_node = Node()
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
			print('Building huffman tree...')
		un_merged_nodes = leaves[:]
		self.nodes = []
		frequencies = [l.frequency for l in leaves]
		# We keep merging 2 least frequency nodes, until only the root node remains. Classic Huffman tree, nothing fancy.
		prev_p = 0
		while len(un_merged_nodes) > 1:
			p = int(100. * (self.nb_classes - len(un_merged_nodes) + 1) / self.nb_classes)
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
			sys.stdout.write('\r100 %')
			print('Huffman tree build complete')
		self.root_node = un_merged_nodes[0]
		self.nodes += [self.root_node]
		self.node_indices = {self.nodes[i]: i for i in range(len(self.nodes))}
		self.node_indices.update({leaves[i]: i for i in range(len(leaves))})
		self.leaves = leaves
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

		if self.verbose:
			print('Setting weights...')

		self.W = self.init((len(self.nodes), input_dim, 1))
		if self.bias:
			self.b = K.zeros((len(self.nodes), 1))
			self.trainable_weights = [self.W, self.b]
		else:

			self.trainable_weights = [self.W]

		self.regularizers = []
		if self.W_regularizer:
			self.W_regularizer.set_param(self.W)
			self.regularizers.append(self.W_regularizer)

		if self.bias and self.b_regularizer:
			self.b_regularizer.set_param(self.b)
			self.regularizers.append(self.b_regularizer)

		if self.activity_regularizer:
			self.activity_regularizer.set_layer(self)
			self.regularizers.append(self.activity_regularizer)

		self.constraints = {}
		if self.W_constraint:
			self.constraints[self.W] = self.W_constraint
		if self.bias and self.b_constraint:
			self.constraints[self.b] = self.b_constraint

		if hasattr(self, 'initial_weights') and self.initial_weights:
			self.set_weights(self.initial_weights)
			del self.initial_weights
		# Class -> path map
		self.class_path_map = K.variable(np.array([[self.node_indices[node] for node in path + [self.root_node] * (self.max_tree_depth - len(path))] for path in self.paths]), dtype='int32')
		super(Huffmax, self).build(input_shape)
		if self.verbose:
			print('Done.')

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
				huffman_code += [
				1]
		return path, huffman_code

	def call(self, x, mask=None):
		input_vector = x[0]
		target_classes = x[1]
		nb_req_classes = self.input_spec[1].shape[1]
		if nb_req_classes is None:
			nb_req_classes = K.shape(target_classes)
		if K.dtype(target_classes) != 'int32':
			target_classes = K.cast(target_classes, 'int32')
		if self.mode == 0:
			# One giant matrix mul
			input_dim = self.input_spec[0].shape[1]
			nb_req_classes = self.input_spec[1].shape[1]
			path_lengths = map(len, self.paths)
			huffman_codes = K.variable(np.array(self.huffman_codes))
			req_nodes = K.gather(self.class_path_map, target_classes)
			req_W = K.gather(self.W, req_nodes)
			y = K.batch_dot(input_vector, req_W, axes=(1, 3))
			if self.bias:
				req_b = K.gather(self.b, req_nodes)
				y += req_b
			y = K.sigmoid(y[:, :, :, 0])
			req_huffman_codes = K.gather(huffman_codes, target_classes)
			return K.prod(req_huffman_codes + y - 2 * req_huffman_codes * y, axis=-1)  # Thug life
		elif self.mode == 1:
			# Many tiny matrix muls
			probs = []
			for i in range(len(self.paths)):
				huffman_code = self.huffman_codes[i]
				path = self.paths[i]
				prob = 1.
				for j in range(len(path)):
					node = path[j]
					node_index = self.node_indices[node]
					p = K.dot(input_vector, self.W[node_index, :, :])[:, 0]
					if self.bias:
						p += self.b[node_index, :][0]
					h = huffman_code[j]
					p = K.sigmoid(p)
					prob *= h + p - 2 * p * h
				probs += [prob]
			probs = K.pack(probs)
			req_probs = K.gather(probs, target_classes)
			req_probs = K.permute_dimensions(req_probs, (0, 2, 1))
			req_probs = K.reshape(req_probs, (-1, nb_req_classes))
			batch_size = K.shape(input_vector)[0]
			indices = arange(batch_size * batch_size, batch_size + 1)
			req_probs = K.gather(req_probs, indices)
			return req_probs

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


class HuffmaxClassifier(Huffmax):
	''' This layer is not differentiable. Hence, can be used for prediction only.
	Train the weights using the Huffmax layer, and transfer them here for prediction.
	For a given 2D input (batch_size, input_dim), outputs a 1D integer array of class labels.
	'''

	def __init__(self, nb_classes, input_dim, **kwargs):
		kwargs['nb_classes'] = nb_classes
		kwargs['input_shape'] = (input_dim,)
		super(HuffmaxClassifier, self).__init__(**kwargs)

	def call(self, x, mask=None):

		def get_node_w(node):
			return self.W[self.node_indices[node], :, :]

		def get_node_b(node):
			return self.b[self.node_indices[node], :]

		def compute_output(input, node=self.root_node):
			if not hasattr(node, 'left'):
				return zeros((K.shape(input)[0],)) + self.node_indices[node]
			else:
				node_output = K.dot(x, get_node_w(node))
				if self.bias:
					node_output += get_node_b(node)
				left_prob = node_output[:, 0]
				right_prob = 1 - node_output[:, 0]
				left_node_output = compute_output(input, node.left)
				right_node_output = compute_output(input, node.right)
				return K.switch(left_prob > right_prob, left_node_output, right_node_output)
		return K.cast(compute_output(x), 'int32')

	def get_output_shape_for(self, input_shape):
		return (input_shape[0],)
