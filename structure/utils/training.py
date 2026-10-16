import torch


def train(net, trainloader, epochs, lr, device):
    """
    Treina o modelo por um número de épocas fornecidas.

    Parâmetros: 
        - net: modelo a ser treinado
        - trainloader: dataset de treinamento dividido em batches
        - epochs: quantidade de rodadas de treinamento
        - lr: learning rate
        - device: serviço disponível (CUDA/GPU)

    Retorna:
        - avg_trainloss: média da loss de treinamento
        - train_accuracy: acurácia do treinamento
    """
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    net.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    avg_trainloss = running_loss / (epochs * len(trainloader))
    train_accuracy = correct / total

    return avg_trainloss, train_accuracy


def test(net, testloader, device):
    """
    Validação do modelo no dataset de validação.

    Parâmetros: 
        - net: modelo a ser treinado
        - testloader: dataset de validação dividido em batches
        - device: serviço disponível (CUDA/GPU)

    Retorna:
        - loss: loss de validação
        - accuracy: acurácia de validação
    """
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy