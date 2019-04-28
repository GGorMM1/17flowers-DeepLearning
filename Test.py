#------------------------------------------------------------
#   coding:utf-8
#------------------------------------------------------------
#	Updata History
#	October  27  16:30, 2018 (Sat) by S.Iwamaru
#------------------------------------------------------------
#
#	テスト
#
#------------------------------------------------------------
import sys
import numpy as np
from PIL import Image
from pathlib import Path
from keras.models import load_model	
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.applications.vgg16 import preprocess_input, decode_predictions

classes = ['Tulip', 'Snowdrop', 'LilyValley', 'Bluebell', 'Crocus',
           'Iris', 'Tigerlily', 'Daffodil', 'Fritillary', 'Sunflower',
           'Daisy', 'ColtsFoot', 'Dandelion', 'Cowslip', 'Buttercup',
           'Windflower', 'Pansy']

"""
	テスト
"""
def main():
	#  コマンドラインより入力(後にparamに変える)
	if len(sys.argv) <= 1:
		quit()				#  画像がないと終了
	
	#  画像のパスを設定
	img_path = Path(".") / sys.argv[1]

	#  画像をarrayとして読み込む
	img = img_to_array( load_img(img_path, target_size=( 224, 224)) )
#	img_ori = array_to_img( img )
#	img_ori.save( "./test_before.png" )	#  PILで保存
	
	#  3次元テンソルを4次元テンソルに変換
	x = np.expand_dims( img, axis=0 )
	x = x / 255.
	
	#  モデル, 重みの読み込み
	#  model_pathの一番最後が最良モデル
	model_path = sorted(list(Path("./model/").glob("*.hdf5")))
	num = len(model_path) - 1
	model = load_model( model_path[num] )
	model.load_weights( model_path[num] )

	print("-------------------------------------------------")
	print("load image:{}".format(img_path))
	print("model_path:{}".format(model_path[num]))
	print("-------------------------------------------------")

	#  テスト
	pred = model.predict(x)[0]

	top = 5
	top_indices = pred.argsort()[-top:][::-1]
	result = [(classes[i], pred[i]) for i in top_indices]
	for x in result:
	    print(x)
	
if __name__ == '__main__':
	main()
