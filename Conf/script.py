import os

old_running_path = "New_Conf/running"
new_running_path = "New_Conf/running_20"

num_epochs = 20

train_losses = []
train_losses_dir = os.path.join(old_running_path, f"train_losses.txt")
new_train_losses_dir = os.path.join(new_running_path, f"train_losses.txt")
test_losses = []
test_losses_dir = os.path.join(old_running_path, f"test_losses.txt")
new_test_losses_dir = os.path.join(new_running_path, f"test_losses.txt")
test_accuracies = []
test_accuracies_dir = os.path.join(old_running_path, f"test_accuracies.txt")
new_test_accuracies_dir = os.path.join(new_running_path, f"test_accuracies.txt")
test_wers = []
test_wers_dir = os.path.join(old_running_path, f"test_wers.txt")
new_test_wers_dir = os.path.join(new_running_path, f"test_wers.txt")

with open(train_losses_dir, 'r') as tl, open(test_losses_dir, 'r') as tl2, open(test_accuracies_dir, 'r') as ta, open(test_wers_dir, 'r') as tw:
    train_losses = tl.read().split(";")
    train_losses = train_losses[:-1]
    with open(new_train_losses_dir, 'w') as f1:
        for i in range(num_epochs):
            f1.write(f"{train_losses[i]};")

    test_losses = tl2.read().split(";")
    test_losses = test_losses[:-1]
    with open(new_test_losses_dir, 'w') as f2:
        for i in range(num_epochs):
            f2.write(f"{test_losses[i]};")

    test_accuracies = ta.read().split(";")
    test_accuracies = test_accuracies[:-1]
    with open(new_test_accuracies_dir, 'w') as f3:
        for i in range(num_epochs):
            f3.write(f"{test_accuracies[i]};")

    test_wers = tw.read().split(";")
    test_wers = test_wers[:-1]
    with open(new_test_wers_dir, 'w') as f4:
        for i in range(num_epochs):
            f4.write(f"{test_wers[i]};")

