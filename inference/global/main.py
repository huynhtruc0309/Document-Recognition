import cv2
import sys
import datetime
import shutil
import torch
import dill
import json
import numpy as np

from pathlib import Path
from train.utils import load_yaml, eval_config
from train.core.metric.evaluate import calculate_embeddings


def cal_accuracy_threshold(embeddings, labels, dataset_len, dataset_name, template_dict):
    # calculate distance
    embeddings = embeddings.unsqueeze(1)
    avg_embedding = template_dict["avg_embedding"].unsqueeze(0)

    distances = torch.sqrt(torch.sum(torch.pow(embeddings - avg_embedding, 2), 2))
    distances = torch.tanh(distances)

    _best_dis, _best_idx = torch.min(distances, dim=1)  # (batch_size,)

    is_error = (_best_idx != labels).detach().cpu().numpy()
    num_error = np.sum(is_error)
    accuracy = (dataset_len - num_error)/dataset_len

    print('='*25, dataset_name, '='*25)
    print('Error ratio:', num_error, '/', dataset_len, '\t Accuracy:', accuracy)

    # plot distance (only train data)
    threshold = {}
    for _class, _class_idx in template_dict['class_to_idx'].items():
        _predict_correctly = torch.logical_and((_best_idx == _class_idx), (_best_idx == labels))
        _distance = _best_dis[_predict_correctly].detach().cpu().numpy()
        # if args["test_dir"].split('/')[-1] == 'train':
        #     plt.plot(_distance, _distance, 'ro')
        #     plt.savefig(os.path.join(os.path.dirname(args["config"]), f'{_class}.png'))
        threshold[_class] = [min(_distance), max(_distance)]
    return threshold


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
    return embedding


def batch_distance(db_embs, input_embbedings):
    input_embbedings = input_embbedings.unsqueeze(1)
    db_embeddings = db_embs.unsqueeze(0)
    return torch.sqrt(torch.sum(torch.pow(db_embeddings - input_embbedings, 2), 2))


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

    # data loader
    train_loader = config["data"]["train"]
    valid_loader = config["data"]["valid"]
    test_loader = config["data"]["test"]

    # model and resume
    model = config["model"]
    model.to(device)
    resume_path = config["inference"]["resume_checkpoint_path"]
    if resume_path != "":
        if Path(resume_path).exists():
            checkpoint = torch.load(resume_path, map_location=device)
            weight = checkpoint["model_state_dict"]
            model.load_state_dict(weight)
        else:
            raise ValueError(
                "Not found resume check point in {} folder!".format(resume_path)
            )

    ###################################################
    #                    Prepare step
    ###################################################

    # calculate average embedding per class
    train_embeddings, train_labels, _ = calculate_embeddings(model, train_loader, device)
    valid_embeddings, valid_labels, _ = calculate_embeddings(model, valid_loader, device)
    test_embeddings, test_labels, _ = calculate_embeddings(model, test_loader, device)

    # load or create dill avg embedding
    template_path = Path(config["inference"]["template_path"])
    if template_path.exists():
        # copy dill file to checkpoint dir
        shutil.copy(template_path, checkpoint_dir)

        # load dill file
        with open(template_path, 'rb') as f:
            template_dict = dill.load(f)
        print('template.dill loaded!')
    else:
        # gen dill file from train_embedding
        template_dict = {
            'class_to_idx': train_loader.dataset.classes,
            'idx_to_class': {val: key for key, val in train_loader.dataset.classes.items()},
            'avg_embedding': [torch.zeros(train_embeddings.shape[0]) for _ in train_loader.dataset.classes]
        }
        for ith, class_idx in enumerate(template_dict['idx_to_class']):
            cur_embs = train_embeddings[train_labels == class_idx]
            avg_emb = torch.mean(cur_embs, dim=0)
            template_dict['avg_embedding'][ith] = avg_emb
        template_dict['avg_embedding'] = torch.stack(template_dict['avg_embedding']).to(torch.device("cpu"))

        # save template
        with open(checkpoint_dir.joinpath(template_path.name), 'wb') as f:
            dill.dump(template_dict, f)
        print('template.dill saved!')

    template_dict['avg_embedding'] = template_dict['avg_embedding'].to(device)

    train_thresh = cal_accuracy_threshold(
        embeddings=train_embeddings,
        labels=train_labels,
        dataset_len=train_loader.dataset.__len__(),
        dataset_name="train",
        template_dict=template_dict
    )
    valid_thresh = cal_accuracy_threshold(
        embeddings=valid_embeddings,
        labels=valid_labels,
        dataset_len=valid_loader.dataset.__len__(),
        dataset_name="valid",
        template_dict=template_dict
    )
    test_thresh = cal_accuracy_threshold(
        embeddings=test_embeddings,
        labels=test_labels,
        dataset_len=test_loader.dataset.__len__(),
        dataset_name="test",
        template_dict=template_dict
    )

    final_thresh = {}
    for _class, _train_thresh in train_thresh.items():
        _valid_thresh = valid_thresh[_class]
        _test_thresh = test_thresh[_class]
        final_thresh[_class] = [
            min(_train_thresh[0], _valid_thresh[0], _test_thresh[0]),
            max(_train_thresh[1], _valid_thresh[1], _test_thresh[1])
        ]
    print(final_thresh)
    template_path = Path(config["inference"]["threshold_path"])
    with open(checkpoint_dir.joinpath(template_path.name), 'w') as f:
        json.dump(str(final_thresh), f)

    ###################################################
    #                    Inference step
    ###################################################
    input_embedding = get_embedding(
        image_path=input_image,
        model=model,
        transforms=train_loader.dataset.post_transforms,
        device=device,
        num_channel=1,
    )
    distances = batch_distance(
        db_embs=template_dict['avg_embedding'],
        input_embbedings=input_embedding
    )
    # get top 1 best fit
    best_distances, best_indices = torch.min(distances, dim=1)
    best_distances = torch.tanh(best_distances)
    item_dis = best_distances.detach().cpu().numpy()[0]
    item_idx = best_indices.detach().cpu().numpy()[0]

    pred_class = template_dict['idx_to_class'][item_idx]

    # check threshold
    lower_bound, upper_bound = final_thresh[pred_class]
    if item_dis <= upper_bound:
        final_predict_class = pred_class
    else:
        final_predict_class = "unknown"
    print("Predict:", final_predict_class)
