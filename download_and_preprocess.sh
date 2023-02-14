wget https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv
wget https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-human-imagelabels.csv
wget https://storage.googleapis.com/openimages/v5/test-annotations-human-imagelabels.csv
wget https://raw.githubusercontent.com/openimages/dataset/master/downloader.py

wget https://nlp.stanford.edu/data/glove.6B.zip

unzip glove.6B.zip -d glove

python preprocess_openimage.py
python downloader.py openimage_id.txt --download_folder=./imgs/ --num_processes=5
python feature_extraction.py
python create_virtual_users.py
