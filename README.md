# TemplateMatching
This project aims to classify image by using template matching method


## Structure of Project
```
TemplateMatching
	├── inference
	|	├── global
	|   	|   ├── configs
	|   	|   └── main.py
	|	└── local
	|   	    ├── configs
	|   	    ├── templates
	|   	    └── main.py
	└── train
		├── configs
		├── core
		├── utils.py
		└── main.py
```


## Training
* Prepare a dataset with structure
    ```
    dataset_name
		├── train
		|	├── class_name1
		|   	|  	└── 1.jpg
		|	└── class_name2
		|       	└── 2.jpg
		├── valid
		|	├── class_name1
		|      	|	└── 3.jpg
		|	└── class_name2
		|       	└── 4.jpg
		└── test
			├── class_name1
			|	└── 5.jpg
			└── class_name2
				└── 6.jpg
    ```

* Create a new config ```.yaml``` file at ```train/configs/``` folder. Ex: ```train/configs/dkx_trainning.yaml```

* Change some parameters:
    * Dataset & Dataloader: [Required]
        - ```dirname```: dataset path
        - ```classes```: training classes
        - ```labels_per_batch```: ***less than*** number of training classes
        - ```samples_per_label```: choose based on memory. But highly recommend that ***shouldn't be greater than*** the smallest number image of a training class.
        - ```batch_size```: labels_per_batch * samples_per_label
        - ```device```: cpu/cuda
    * Model: [Optional]
        - Choose a backbone for getting the best result on your dataset. Ex: ```train/core/model/models.py```
    * Loss: [Optional]
    * Optim: [Optional]
    * Trainer: [Required]
        - ```max_epoch```: default 1000
        - ```device```: cpu/cuda
        - ```checkpoint_dir```
        - ```resume_checkpoint_path```: 
        - ```valid_frequency```: only run validation after *x* epoch
        - ```early_stoping```: stop training if after *valid_frequency * early_stoping* epochs, model can't get better 
* Running: At root project 
    ```sh
    python /train/main.py your_config_path.yaml
    ```
## Inference
There are 2 inference types:
* ***Global inference***: Using average embedding vector of each class to check distance and decide the class of new input image

* ***Local inference***: Using some protential image of each class instead of a average embedding

You can choose global or local inference type. It's depend on you. We recommend using global inference. But in some cases, local inference is the optimal choice.

* Require
	* Create Dataset / Template folder as same as structure of training dataset.
	* Create a new config ```.yaml``` file at ```inference/(global|local)/configs/``` folder
	* Change some parameter if you need.

### Global Inference
* Flow
![inference_flow](https://user-images.githubusercontent.com/38365552/130029555-f945c4b5-bc2a-4aca-a8b1-5df2b08684bf.png)
* Running: At root project
	```sh
	PYTHONPATH=. python inference/global/main.py inference/global/configs/dkx_global_inference.yaml your_test_image_path
	```

### Local Inference
* Flow
![local_inference_flow](https://user-images.githubusercontent.com/38365552/130032319-44cfc79d-cee1-4197-bb99-8fbb41534be3.png)
* Running: At root project
	```sh
	PYTHONPATH=. python inference/local/main.py inference/global/configs/dkx_local_inference.yaml your_test_image_path
	```