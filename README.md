# Cityscapes
Image detection of common urban street objects

## Running the Flask Application 

Run the **app.py** file from the **training_demo/webapp/** directory. 

`python app.py` 

Go to **localhost:5000** to access the application from the browser of your choice.

## Structure of the folder

```

|   Data Preprocessing Update.pdf
|   Data Selection Proposal.pdf
|   Model Update (Deliverable 3).pdf
|   README.md
|   tree.txt
|   
\---training_demo
    |   
    +---images
    |   +---test
    |   |       frankfurt[].png
    |   |       frankfurt[].png.json
    |   |       
    |   \---train
    |           aachen[].png
    |           aachen[].png.json
    |           bochum[].png
    |           bochum[].png.json
    |           bremen[].png
    |           bremen[].png.json
    |           
    +---scripts
    |       exporter_main_v2.py
    |       generate_tfrecord.py
    |       JSON_to_CSV.py
    |       model_main_tf2.py
    |       Preprocessing_datasets.JSON
    |       
    \---webapp
        |   app.py
        |   cityscapes_predictor.py
        |   
        +---annotations
        |       label_map.pbtxt
        |       test_labels.csv
	|       train_labels.csv
        |       
        +---exported-models
        |   |   saved_model.pb
        |   |   
        |   +---assets
        |   \---variables
        |           .gitattributes
        |           variables.data-00000-of-00001
        |           variables.index
        |           
        +---templates
        |       index.html
        |       
        \---__pycache__
                cityscapes_predictor.cpython-37.pyc

```


