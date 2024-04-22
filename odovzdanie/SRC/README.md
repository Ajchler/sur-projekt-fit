# SUR projekt

Authors: Vojtech Eichler (xeichl01), Adam Zvara (xzvara01)<br>
Date: 04/2024

# How to run

Install requirements:
```
pip install -r requirements.txt
```

Install the sox library:
```
sudo apt install libsox-dev
```

Training the audio model (the model is stored in `audio_classifier.pkl`):
```
python train_audio_gmm.py
```

Training the image model (the model is stored in `image_classifier.pkl`):
```
python train_image_cnn.py
```

Running evaluation:
```
python evaluate.py --audio
python evaluate.py --image
```
