from RNN import RNNModel
import torch.optim as optim
from torch import nn
import torch
from data_utils import load_data, generate_combinations, NewThread
from model_utils import train_net, test_net


def train_test(lr, hidden_dim, batch_size, num_epochs):
    print(f'*******params:({lr},{hidden_dim},{batch_size},{num_epochs})*********')
    best_test_acc = 0
    X_train, train_loader, test_loader = load_data(batch_size)
    in_dim = X_train.shape[1]
    out_dim = 5

    model = RNNModel(in_dim, hidden_dim, out_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_net(model, train_loader, optimizer, criterion, device)
        test_acc = test_net(model, test_loader, device)
        best_test_acc = max(test_acc, best_test_acc)
        if epoch % 100 == 0 or epoch == num_epochs - 1:
            print(f'Epoch: {epoch},\t train_loss: {train_loss:.4f},\ttrain_acc: {train_acc},\ttest_acc: {test_acc}')

    return best_test_acc


def search_parameters(params):
    lr, hidden_dim, batch_size, num_epochs = params
    return train_test(lr, hidden_dim, batch_size, num_epochs)


def main():
    hidden_dim_list = [16,32, 64, 128, 256, 512, 1024]
    num_epochs_list = [20, 100, 200, 500]
    learning_rates_list = [0.1,0.001, 1e-4, 1e-5]
    batch_size_list = [2,32, 64, 128, 256, 1024]
    combinations = generate_combinations(learning_rates_list, hidden_dim_list, batch_size_list, num_epochs_list)
    best_acc = 0
    for i in range(0, len(combinations), 4):
        threads = [NewThread(target=search_parameters, args=(params,)) for params in combinations[i:i + 4]]
        for thread in threads:
            thread.start()
        for thread in threads:
            test_acc = thread.join()
            best_acc = max(best_acc, test_acc)

    print(f'best_acc:{best_acc}')


if __name__ == '__main__':
    main()
