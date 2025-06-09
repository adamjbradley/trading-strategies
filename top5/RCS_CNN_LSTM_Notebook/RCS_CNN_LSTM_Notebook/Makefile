.PHONY: setup run notebook export clean docker-build docker-run

setup:
	conda env create -f environment.yml || conda env update -f environment.yml

run:
	python RCS_CNN_LSTM.ipynb

notebook:
	jupyter notebook RCS_CNN_LSTM.ipynb

export:
	python inference.py --export

clean:
	rm -rf __pycache__ *.onnx *.h5 results/ models/

docker-build:
	docker build -t rcs_cnn_lstm .

docker-run:
	docker run -p 8888:8888 rcs_cnn_lstm
