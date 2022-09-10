import sys
import datetime
import shutil
import torch
import dill
import cv2
import numpy as np

from pathlib import Path
from train.utils import load_yaml, eval_config
from train.core.metric.evaluate import calculate_embeddings


def get_embedding(image_path, model, transforms, device, num_channel):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for transform in transforms:
        image = transform(image=image)
    # gray scale image
    if num_channel == 1:
        image = np.expand_dims(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), axis=-1)

    image_tensor = torch.from_numpy(image)
    image_tensor = torch.div(image_tensor, 255.0)
    image_tensor = image_tensor.permute(2, 0, 1).contiguous()
    image_tensor = image_tensor.unsqueeze(dim=0)  # [1 x H x W] -> [1 x 1 x H x W]
    image_tensor = image_tensor.to(device, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        embedding = model(image_tensor)
        embedding = embedding.detach().cpu().numpy()
        return embedding


def calculate_distance(embedding_1, embedding_2):
    distance = embedding_1 - embedding_2
    distance = float(np.sqrt(np.sum(distance**2)))
    return distance


def get_topk(image_path, database, model, transforms, device, num_channel, topk=1):
    embedding = get_embedding(image_path, model, transforms, device, num_channel)

    # For each template in database, calculate distance from input image to that template
    distances = []
    for db_image_path, db_embedding in database.items():
        distance = calculate_distance(embedding, db_embedding)
        distances.append((db_image_path, distance))

    # Sort templates from smallest distance to largest distance
    distances.sort(key=lambda x: x[1])
    print(distances)

    # Get top-k template that are most similar to input image
    output = []
    for i in range(topk):
        rank = i + 1
        output.append(
            {
                "rank": rank,
                "template_path": distances[i][0],
                "distance": distances[i][1]
            }
        )
    return output


if __name__ == "__main__":
    config_path = sys.argv[1]
    input_image = sys.argv[2]

    # load config
    config = load_yaml(config_path)
    config = eval_config(config)

    # init checkpoint dir and copy config file to this folder
    checkpoint_dir = Path(
        config["inference"]["checkpoint_dir"],
        datetime.datetime.now().strftime("%Y%d%m_%H%M%S"),
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, checkpoint_dir)

    # device
    device = config["inference"]["device"]

    # dataset
    template_loader = config["template_loader"]

    # model and resume
    model = config["model"]
    model.to(device)
    resume_path = config["inference"]["resume_checkpoint_path"]
    if resume_path != "":
        if Path(resume_path).exists():
            checkpoint = torch.load(resume_path, map_location=device)
            weight = checkpoint["model_state_dict"]
            model.load_state_dict(weight)
            print("Resumed checkpoint!")
        else:
            raise ValueError(
                "Not found resume check point in {} folder!".format(resume_path)
            )

    # load or create database.dill
    database_path = Path(config["inference"]["database_path"])
    if database_path.exists():
        # copy dill file to checkpoint dir
        shutil.copy(database_path, checkpoint_dir)

        # load dill file
        with open(database_path, 'rb') as f:
            database_dict = dill.load(f)
        print('database.dill loaded!')
    else:
        # gen database from template_dataset
        embeddings, labels, image_paths = calculate_embeddings(model, template_loader, device)
        database_dict = {}
        for ith, image_path in enumerate(image_paths):
            embedding = embeddings[ith]
            database_dict[image_path] = embedding.detach().cpu().numpy()

        # save template
        with open(checkpoint_dir.joinpath(database_path.name), 'wb') as f:
            dill.dump(database_dict, f)
        print('database.dill saved!')

    topk = get_topk(
        image_path=input_image,
        database=database_dict,
        model=model,
        transforms=template_loader.dataset.post_transforms,
        device=device,
        num_channel=template_loader.dataset.num_channel,
        topk=1
    )
    print(topk)
