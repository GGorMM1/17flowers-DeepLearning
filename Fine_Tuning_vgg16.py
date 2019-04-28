#------------------------------------------------------------
#   coding:utf-8
#------------------------------------------------------------
#	Updata History
#	October  30  16:00, 2018 (Tue) by S.Iwamaru
#------------------------------------------------------------
#
#	転移学習を行うプログラム(VGG16)
#
#------------------------------------------------------------

import math, argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.callbacks import CSVLogger, EarlyStopping, TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

#  parameter
lr = 1e-4					#  SGDのlr係数
momentum = 0.9				#  SGDのmomentum係数
input_shape = ( 224, 224, 3)	#  VGG16のinput_shape

classes = ['Tulip', 'Snowdrop', 'LilyValley', 'Bluebell', 'Crocus',
           'Iris', 'Tigerlily', 'Daffodil', 'Fritillary', 'Sunflower',
           'Daisy', 'ColtsFoot', 'Dandelion', 'Cowslip', 'Buttercup',
           'Windflower', 'Pansy']

"""
	モデル(VGG16の転移学習)
"""
class VGG16_Transfer(object):
	def __init__(self, args):
		self.train_dir = args.train_dir
		self.validation_dir = args.validation_dir
		self.batch_size = args.batch_size
		self.epochs = args.epochs
				
		#  VGG16の読み込み
		vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
		#vgg16.summary()
	
		#  モデル定義
		self.model = self.build_transfer_model(vgg16)
		#  モデル構築
		self.model.compile( optimizer=SGD(lr=lr, momentum=momentum),
						    loss='categorical_crossentropy',
					   		metrics=["accuracy"] )

		#  モデルの表示
		print("Model Summary------------------")
		self.model.summary()

	"""
		モデル構築
	"""
	def build_transfer_model( self, vgg16 ):
		model = Sequential(vgg16.layers)
		#  再学習しないように指定
		for layer in model.layers[:15]:
			layer.trainable = False
		#  FC層を構築
		model.add(Flatten())
		model.add(Dense(256, activation="relu"))
		model.add(Dropout(0.5))
		model.add(Dense(len(classes), activation="softmax"))
		
		return model
	
	"""
		モデル学習
	"""
	def training( self ):
		#  学習データ
		train_datagen = ImageDataGenerator(
				rescale=1/255.,
				rotation_range=90,				#  画像をランダムに回転
				shear_range=0.1,				#  せん断する
				zoom_range=0.1,					#  ランダムにズーム
				horizontal_flip=True,			#  水平方向にランダム反転
				vertical_flip=True,				#  垂直方向にランダム反転
				preprocessing_function=self.preprocess_input	
		)	

		train_generator = train_datagen.flow_from_directory(
				self.train_dir,
				target_size=input_shape[:2],
				classes=classes,
				batch_size=self.batch_size
		)

		#  検証データ
		test_datagen = ImageDataGenerator(
				rescale=1/255.,
#				rotation_range=90,				#  画像をランダムに回転
#				shear_range=0.1,				#  せん断する
#				zoom_range=0.1,					#  ランダムにズーム
#				horizontal_flip=True,			#  水平咆哮にランダム反転
#				vertical_flip=True,				#  垂直方向にランダム反転
				preprocessing_function=self.preprocess_input	
		)	

		validation_generator = test_datagen.flow_from_directory(
				self.validation_dir,
				target_size=input_shape[:2],
				classes=classes,
				batch_size=self.batch_size
		)

		#  steps_per_epoch, validation_stepsの算出
		steps_per_epoch = math.ceil( train_generator.samples / self.batch_size )
		validation_steps = math.ceil( validation_generator.samples / self.batch_size )
	
		#  callbacksの設定
		csv_logger = CSVLogger("./training.log")
#		early_stop = EarlyStopping( monitor="val_loss", mode="auto" )
		Path("./logs").mkdir(parents=True, exist_ok=True)
		tensor_board = TensorBoard( "./logs",
									 histogram_freq=0,
									 write_graph=True,
									 write_images=True )
		Path("./model").mkdir(parents=True, exist_ok=True)	
		check_point = ModelCheckpoint( filepath='./model/model.{epoch:02d}-{val_loss:.4f}.hdf5',
									   monitor="val_loss",
									   save_best_only=True,
#									   save_weights_only=True,
									   mode="auto" )
						    
		cb = [ csv_logger, tensor_board, check_point ]

		#  学習
		hist = self.model.fit_generator( 
				  	train_generator,
		 	  	  	steps_per_epoch=steps_per_epoch,
		 	  	  	epochs=self.epochs,
				  	validation_data=validation_generator,
				  	validation_steps=validation_steps,				  
				  	callbacks=cb
		)
		
		return hist

	# keras.applications.imagenet_utilsのxは4Dテンソルなので
	# 3Dテンソル版を作成
	def preprocess_input(self,x):
	    """Preprocesses a tensor encoding a batch of images.
	    # Arguments
	        x: input Numpy tensor, 3D.
	    # Returns
	        Preprocessed tensor.
	    """
	    # 'RGB'->'BGR'
	    x = x[:, :, ::-1]
	    # Zero-center by mean pixel
	    x[:, :, 0] -= 103.939
	    x[:, :, 1] -= 116.779
	    x[:, :, 2] -= 123.68
	    return x
		
	"""
		損失,精度のグラフ描画
	"""
	def plot_history( self, hist ):
		#print(history.history.keys())
		
		#  損失の経過をプロット
		plt.figure()
		loss = hist.history['loss']
		val_loss = hist.history['val_loss']
		plt.plot( range(self.epochs), loss, marker='.', label='loss' )
		plt.plot( range(self.epochs), val_loss, marker='.', label='val_loss' )
		plt.legend( loc='best', fontsize=10 )
		plt.xlabel('epoch')
		plt.ylabel('loss')
		plt.title('model loss')
		plt.legend( ['loss','val_loss'], loc='upper right')
		plt.savefig( "./model_loss.png" )
	#	plt.show()
	
		#  精度の経過をプロット
		plt.figure()
		acc = hist.history['acc']
		val_acc = hist.history['val_acc']
		plt.plot( range(self.epochs), acc, marker='.', label='acc' )
		plt.plot( range(self.epochs), val_acc, marker='.', label='val_acc' )
		plt.legend( loc='best', fontsize=10 )
		plt.xlabel('epoch')
		plt.ylabel('acc')
		plt.title('model accuracy')
		plt.legend( ['acc','val_acc'], loc='lower right')
		plt.savefig( "./model_accuracy.png" )
	#	plt.show()
	
"""
	メイン処理
"""
if __name__ == '__main__':
	#  parameter
	parser = argparse.ArgumentParser(description="17flowers Fine Tuning ")
	parser.add_argument("--train_dir", default= "./train_images/")
	parser.add_argument("--validation_dir", default="./test_images/")
	parser.add_argument("--batch_size", "-b", type=int, default=32,
						help="Number of images in each mini-batch")
	parser.add_argument("--epochs", "-e", type=int, default=100,
						help="Number of sweeps over the dataset to train")
	parser.add_argument("--img_size", "-s", type=int, default=224,
						help="Number of images size")
	args = parser.parse_args()
	
	#  モデル
	md = VGG16_Transfer(args)
	#  学習
	print("Training Start------------------")
	hist = md.training()
	
	#  グラフの表示
	md.plot_history( hist )

