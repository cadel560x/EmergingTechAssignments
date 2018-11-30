# Adapted from https://github.com/ianmcloughlin/jupyter-teaching-notebooks/blob/master/mnist.ipynb
import matplotlib.pyplot as plt

import sklearn.preprocessing as pre
import numpy as np
import keras as kr
import gzip

from keras.models import load_model
from PIL import Image

#
with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
		test_lbl = f.read()
test_lbl = np.array(list(test_lbl[ 8:])).astype(np.uint8)
encoder = pre.LabelBinarizer()
encoder.fit(test_lbl)

# 
model = None

def load():
	global model
	filename = input("Please enter a HDF5 file to load: ")
	model = load_model(filename)
	# DEBUG print(type(model))
	model.summary()

def configure():
	global model
	if model:
		confirmation = input("\nThe current model is about to be destroyed. Continue? (y/n) ")
		if confirmation == "y":
			del model
		elif confirmation == "n":
			return
	
	model = kr.models.Sequential()
	
	print("Model options")
	
	input_str = input("\nFirst layer: how many input neurons? (default: 784) ")
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
	
	print("Second layer: which activation function to use? (e.g. linear, sigmoid, elu, selu, relu, softplus, softmax)  ")
	print("More activation functions at https://keras.io/activations/")
	activation_function = input()
	
	# First and second layers
	model.add(kr.layers.Dense(units=neurons, activation=activation_function, input_dim=inital_neurons))
	
	answer = input("\nAdd another layer? (y/n) ")
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
		
		print("New layer: which activation function to use? (e.g. linear, sigmoid, elu, selu, relu, softplus, softmax)  ")
		print("More activation functions at https://keras.io/activations/")
		activation_function = input()
		
		model.add(kr.layers.Dense(units=neurons, activation=activation_function))
		
		answer = input("\nAdd another layer? (y/n) ")
			
	# Add a hidden layer with 1000 neurons and an input layer with 784.	
	# model.add(kr.layers.Dense(units=600, activation='linear', input_dim=784))
	# global model.add(kr.layers.Dense(units=600, activation='linear'))
	# model.add(kr.layers.Dense(units=400, activation='relu'))
	# Add a three neuron output layer.
	# model.add(kr.layers.Dense(units=10, activation='softmax'))
	# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	print("Last layer: it is set by default to 10 output neurons strictly")
	print("Last layer: which activation function to use? (e.g. linear, sigmoid, elu, selu, relu, softplus, softmax) ")
	print("More activation functions at https://keras.io/activations/")
	activation_function = input()
	model.add(kr.layers.Dense(units=10, activation=activation_function))
	
	print("\nCompile options")
	print("Which loss function to use? (e.g. binary_crossentropy, categorical_crossentropy, mse, mae, mape, msle, kld, cosine) ")
	print("More loss functions at https://github.com/keras-team/keras/blob/master/keras/losses.py")
	loss_function = input()
	
	print("\nWhich optimizer? (e.g sgd, rmsprop, adam, adadelta, adagrad)")
	print("More optimizers at https://keras.io/optimizers/")
	optimizer_value = input()
	model.compile(loss=loss_function, optimizer=optimizer_value, metrics=['accuracy'])
	

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
	outputs = encoder.transform(train_lbl)

	# DEBUG print("outputs", outputs, outputs.shape)
	
	model.summary()
	
	epoch_input = 2
	input_str = input("Train model: how many epochs? (default: 2) ")
	if not len(input_str) == 0:
		try:
			epoch_input = int(input_str)
			if epoch_input <= 0:
				raise ValueError('InvalidInput')
		except ValueError:
			# handle input error or assign default for invalid input
			print('Second layer neurons can\'t be less or equal to zero')
			
	batch_input = 100
	input_str = input("Train model: batch size? (default: 100) ")
	if not len(input_str) == 0:
		try:
			batch_input = int(input_str)
			if batch_input <= 0:
				raise ValueError('InvalidInput')
		except ValueError:
			# handle input error or assign default for invalid input
			print('Second layer neurons can\'t be less or equal to zero')
			
	
	model.fit(inputs, outputs, epochs=epoch_input, batch_size=batch_input)
	
	print("\nSave this model into a HDF5 file? (y/n) ")
	save_file = input()
	
	if save_file == "y":
		save()
	
def test():
	global model
	global encoder
	global test_lbl
	
	if not model:
		print("Empty model. Please create/load a model first")
		return
	
	with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
		test_img = f.read()
		
	test_img = ~np.array(list(test_img[16:])).reshape(10000, 784).astype(np.uint8) / 255.0
	# local_test_lbl =  np.array(list(test_lbl[8:])).astype(np.uint8)
	
	# encoder.fit(local_test_lbl)
	
	model.summary()
	
	# DEBUG model.predict(test_img)
	
	rs = (encoder.inverse_transform(model.predict(test_img)) == test_lbl).sum()
	pct = (rs/10000)*100
	print("\nModel has made", rs, "successful predictions out of 10000 tests (", pct, "%)")
	

def save():
	global model
	# Save model
	if not model:
		print("There is no model!\nPlease create/load a model first")
		return

	filename = input("Please enter a filename: ")
	model.save(filename)
	
def png_read():
	if not model:
		print("There is no model!\nPlease create/load a model first")
		return
	
	filename = input("Please enter a PNG image file: ")
	img = Image.open(filename).convert("L")
	
	print("Image width (pixels): ", img.size[0], " Image height (pixels): ", img.size[1])
	print("\n!Notice! Processing width times processing height must equal the amount of input neurons of a model!\n")
	
	proc_width = img.size[0]
	proc_height = img.size[1]
	
	input_str = input("Please enter new image processing width: (Press enter to keep original dimension) ")
	if input_str:
		try:
			proc_width = int(input_str)
		except ValueError:
				# handle input error or assign default for invalid input
				print('Invalid input')
				
	input_str = input("Please enter new image processing height: (Press enter to keep original dimension) ")
	if input_str:
		try:
			proc_height = int(input_str)
		except ValueError:
				# handle input error or assign default for invalid input
				print('Invalid input')
	
	if (proc_width != img.size[0]) or (proc_height != img.size[1]):
		# img = img.resize((proc_width,proc_height), Image.ANTIALIAS)
		img.thumbnail((proc_width,proc_height), Image.ANTIALIAS)
		
	print("\nProcessing width:", proc_width, "Processing height:", proc_height)
	
	# DEBUG
	# print("img.size",img.size)
	# plt.imshow(img)
	# plt.show()
	
	one_dim =  proc_width*proc_height
	
	im2arr = np.array(img.getdata())
	# DEBUG print(im2arr.shape)
	
	# im2arr = np.array(img).reshape(1,784)
	# im2arr = np.array(img).reshape(1,one_dim)
	im2arr = np.array(list(im2arr)).reshape(1, one_dim).astype(np.uint8) / 255.0
	# DEBUG print(im2arr)
	# DEBUG print(im2arr.shape)
	
	pred = model.predict(im2arr)
	# DEBUG print(pred)
	
	rs = encoder.inverse_transform(pred)
	# DEBUG print(rs)

	print("The program predicts that the image is a:", rs)


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