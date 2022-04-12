### website
- install `flask` before you run the app.py
- download weight file and extract it in /saved_model
  - https://drive.google.com/file/d/1T1NnlwbbKJgHGNJ4f5oUCiPuPfW5-b8J/view?usp=sharing
- `python app.py` to run
- the prediction is given by our pretrained VGG and AlexNet model
- choose any image (cat or dog) to make a prediction!
    
### dataset
- you can download the dataset at <https://www.kaggle.com/c/dogs-vs-cats>
- unzip the data ant put them in `732Project/data`

### train
- `python train.py` to train the model
- If you are training modified VGG, change size to 128*128 in `transform_function()` in `utility.py`.


