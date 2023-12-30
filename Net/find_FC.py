import torch
import torch.nn as nn
import torch.optim as optim
from AutoFCnet import AutoFCNet
from model_utils import train_net, test_net
from data_utils import load_data, generate_combinations, NewThread


def train_test(lr, batch_size, num_epochs, hidden_sizes):
    print(f'*******params:({lr},{batch_size},{num_epochs},{hidden_sizes})*********')
    best_test_acc = 0
    X_train, train_loader, test_loader = load_data(batch_size)
    input_size = X_train.shape[1]
    num_classes = 5

    model = AutoFCNet(input_size, num_classes, hidden_sizes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_net(model, train_loader, optimizer, criterion, device)
        test_acc = test_net(model, test_loader, device)
        best_test_acc = max(test_acc, best_test_acc)
        if epoch % 100 == 0 or epoch == num_epochs - 1:
            print(f'Epoch: {epoch}/{num_epochs},\t train_loss: {train_loss:.4f},\ttrain_acc: {train_acc},\ttest_acc: {test_acc}')

    return best_test_acc


def search_parameters(params):
    lr, batch_size, num_epochs, hidden_sizes = params
    return train_test(lr, batch_size, num_epochs, hidden_sizes)


def main():
    lr = [0.1, 0.0001, 1e-4, 1e-5]
    batch_size = [2, 32, 64, 128, 256, 1024]
    num_epochs = [20, 100, 200, 500]
    hidden_sizes = [
        [32, 64],
        [40,80,40],
        [40, 80, 160, 80, 40],
        [32, 64, 128, 128, 32],
        # [32, 64, 128, 256, 512, 1024, 1024, 512, 256, 128, 64, 32],
        # [32, 64, 128, 256, 256, 256, 256, 128, 64, 32]
    ]
    combinations = generate_combinations(lr, batch_size, num_epochs, hidden_sizes)
    best_acc = 0
    num_thread = 8
    for i in range(0, len(combinations), num_thread):
        threads = [NewThread(target=search_parameters, args=(params,)) for params in combinations[i:i + num_thread]]
        for thread in threads:
            thread.start()
        for thread in threads:
            test_acc = thread.join()
            best_acc = max(best_acc, test_acc)

    print(f'best_acc:{best_acc}')


if __name__ == '__main__':
    main()
