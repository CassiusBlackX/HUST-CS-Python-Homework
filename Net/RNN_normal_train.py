from RNN import RNNModel
from model_utils import train_net, test_net
from data_utils import load_data
from torch import nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt


def main():
    X_train, train_loader, test_loader = load_data(32)
    in_dim = X_train.shape[1]
    out_dim = 5
    model = RNNModel(in_dim, 32, out_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    num_epochs = 200
    train_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(num_epochs):
        train_loss, train_acc = train_net(model, train_loader, optimizer, criterion, device)
        test_acc = test_net(model, test_loader, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    # 绘制图表
    plt.figure(figsize=(12, 4))

    # 绘制训练损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.legend()

    # 绘制训练准确率和测试准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accs, label='Train Acc')
    plt.plot(range(1, num_epochs + 1), test_accs, label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
