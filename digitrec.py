# Adapted from https://github.com/ianmcloughlin/jupyter-teaching-notebooks/blob/master/mnist.ipynb

import sklearn.preprocessing as pre
import numpy as np
import keras as kr
import gzip

from keras.models import load_model
from PIL import Image

encoder = pre.LabelBinarizer()
# model = kr.models.Sequential()
model = None

def load():
	global model
	filename = input("Please enter a HDF5 file to load: ")
	model = load_model(filename)
	# print(type(model))
	model.summary()

def configure():
	global model
	if model:
		confirmation = input("The current model is about to be destroyed. Continue? (y/n) ")
		if confirmation == "y":
			del model
		elif confirmation == "n":
			return
	
	model = kr.models.Sequential()
	
	input_str = input("First layer: how many input neurons? (default: 784) ")
	inital_neurons = 784
	if not len(input_str) == 0:
		try:
			inital_neurons = int(input_str)
			if inital_neurons <= 0:
				raise ValueError('InvalidInput')
		except ValueError:
			# handle input error or assign default for invalid input
			print('First layer neurons can\'t be less or equal to zero')

	
	neurons = 600
	input_str = input("Second layer: how many neurons? (default: 600) ")
	if not len(input_str) == 0:
		try:
			neurons = int(input_str)
			if neurons <= 0:
				raise ValueError('InvalidInput')
		except ValueError:
			# handle input error or assign default for invalid input
			print('Second layer neurons can\'t be less or equal to zero')
	
	input_str = input("Second layer: which activation function to use? (linear, sigmoid, relu, softmax) ")
	activation_function = input_str
	
	# First and second layers
	model.add(kr.layers.Dense(units=neurons, activation=activation_function, input_dim=inital_neurons))
	
	answer = input("Add another layer? (y/n) ")
	while answer == "y":
		neurons = 400
		input_str = input("New layer: how many neurons? (default: 400) ")
		if not len(input_str) == 0:
			try:
				neurons = int(input_str)
				if neurons <= 0:
					raise ValueError('InvalidInput')
			except ValueError:
				# handle input error or assign default for invalid input
				print('New layer neurons can\'t be less or equal to zero')
		
		input_str = input("New layer: which activation function to use? (linear, sigmoid, relu, softmax) ")
		activation_function = input_str
		
		model.add(kr.layers.Dense(units=neurons, activation=activation_function))
		
		answer = input("Add another layer? (y/n) ")
			
	# Add a hidden layer with 1000 neurons and an input layer with 784.	
	# model.add(kr.layers.Dense(units=600, activation='linear', input_dim=784))
	# global model.add(kr.layers.Dense(units=600, activation='linear'))
	# model.add(kr.layers.Dense(units=400, activation='relu'))
	# Add a three neuron output layer.
	# model.add(kr.layers.Dense(units=10, activation='softmax'))
	
	print("Last layer: it is set by default to 10 output neurons strictly")
	input_str = input("Last layer: which activation function to use? (linear, sigmoid, relu, softmax) ")
	activation_function = input_str
	model.add(kr.layers.Dense(units=10, activation=activation_function))
	
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	
# def compile():
	# global model
	# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

def train():
	global model
	global encoder
	
	if not model:
		print("Empty model. Please create/load a model first")
		return
	
	with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f:
		train_img = f.read()

	with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f:
		train_lbl = f.read()
		
	train_img = ~np.array(list(train_img[16:])).reshape(60000, 28, 28).astype(np.uint8) / 255.0
	train_lbl =  np.array(list(train_lbl[ 8:])).astype(np.uint8)

	inputs = train_img.reshape(60000, 784)

	# encoder = pre.LabelBinarizer()
	encoder.fit(train_lbl)
	outputs = encoder.transform(train_lbl)

	model.summary()
	
	model.fit(inputs, outputs, epochs=2, batch_size=100)
	
	# print(type(model))
	print("\nSave this model into a HDF5 file? (y/n) ")
	save_file = input()
	
	if save_file == "y":
		save()
	
def test():
	global model
	global encoder
	
	if not model:
		print("Empty model. Please create/load a model first")
		return
	
	with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
		test_img = f.read()

	with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
		test_lbl = f.read()
		
	test_img = ~np.array(list(test_img[16:])).reshape(10000, 784).astype(np.uint8) / 255.0
	test_lbl =  np.array(list(test_lbl[ 8:])).astype(np.uint8)
	
	encoder.fit(test_lbl)
	# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	model.summary()
	# model.build()
	# model.summary()
	
	# model.predict(test_img)
	
	rs = (encoder.inverse_transform(model.predict(test_img)) == test_lbl).sum()
	pct = (rs/10000)*100
	print("\nModel has made", rs, "successful predictions out of 10000 tests (", pct, "%)")
	
	# model.summary()

def save():
	global model
	# Save model
	filename = input("Please enter a filename: ")
	model.save(filename)
	
def read_png():
	filename = input("Please enter a PNG image file: ")
	img = Image.open(filename).convert("L")
	img = img.resize((28,28))
	im2arr = np.array(img)
	im2arr = im2arr.reshape(1,28,28,1)
	print(im2arr)


choice = True
while choice:
	print("""
	1. Load model from HDF5 file
	2. Create, configure and compile model
	3. Train with MNIST training images
	4. Test against the MNIST testing images
	5. Save model
	6. Read and predict from a PNG file
	7. Exit
	""")
	choice = input("Option: ")
	
	if choice == "1":
		load()
	elif choice =="2":
		configure()
	elif choice =="3":
		train()
	elif choice=="4":
		test()
	elif choice =="5":
		save()
	elif choice =="6":
		png_read()
	elif choice=="7":
		choice = None
	else:
		print("Invalid option")