
 #Loads to torch tensors
import torch


class KerasLoader:
    #Use ONLY train data 
    def __init__(self, X, y):
        #converts x and y to numpy arr so they can be torch tensor
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            X = X.to_numpy()
            y = y.to_numpy()
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()
    #getters and setters
    def get_trainloader(dataset):
        return torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)
    def get_testloader(dataset):
        return torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)
    #to get lenght, for enumerator use
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]