import torch

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


@torch.no_grad()
def calculate_test_loss(model, test_loader, loss_function, epoch, device, writer):
    model.eval()

    test_loss = 0.0
    for data in test_loader:
        anchor, positive, negative = data

        anchor = anchor.to(device, non_blocking=True)
        positive = positive.to(device, non_blocking=True)
        negative = negative.to(device, non_blocking=True)

        anchor_embedding = model(anchor)
        positive_embedding = model(positive)
        negative_embedding = model(negative)

        loss = loss_function(anchor_embedding, positive_embedding, negative_embedding)
        test_loss += loss.item()

    test_loss = test_loss / len(test_loader.dataset)
    writer.add_scalar("test/loss", test_loss, epoch)
    return test_loss


def calculate_test_accuracy(model, test_loader, train_loader, epoch, device, writer):
    model.eval()
    # Set sampler to validating mode to sample images and labels sequentially
    train_loader.sampler.sequential_sampling = True
    # Set test loader in train mode to sample images and labels sequentially
    test_loader.dataset.return_triplets = False

    # Calculate all embeddings of training set and testing set
    embeddings_train, labels_train, _ = calculate_embeddings(model, train_loader, device)
    embeddings_val, labels_val, _ = calculate_embeddings(model, test_loader, device)

    accuracy = accuracy_KNN(embeddings_train, labels_train, embeddings_val, labels_val)
    accuracy = accuracy * 100
    writer.add_scalar("test/accuracy", accuracy, epoch)

    # Set to default value
    train_loader.sampler.sequential_sampling = False
    test_loader.dataset.return_triplets = True

    return accuracy


@torch.no_grad()
def calculate_embeddings(model, loader, device):
    model.eval()

    embeddings = []
    labels = []
    image_paths = []
    for data in loader:
        images = data[0].to(device, non_blocking=True)
        labels_ = data[1].to(device, non_blocking=True)

        embedding = model(images)
        embeddings.append(embedding)
        labels.append(labels_)
        image_paths.extend(list(data[2]))

    embeddings = torch.cat(embeddings, dim=0)  # shape: [N x embedding_size]
    labels = torch.cat(labels, dim=0)  # shape: [N]

    return embeddings, labels, image_paths


def accuracy_KNN(embeddings_train, labels_train, embeddings_val, labels_val):
    # Construct training set
    X_train = embeddings_train.cpu().numpy()
    y_train = labels_train.cpu().numpy()

    # Construct a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=1, p=2, n_jobs=-1)
    knn.fit(X_train, y_train)

    # Construct validating set
    X_val = embeddings_val.cpu().numpy()
    y_val = labels_val.cpu().numpy()

    # Find the nearest sample in training set to a validating image
    y_pred = knn.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    return accuracy
