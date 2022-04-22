import torch


DTYPE = torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class NeuralNet:
    def __init__(self, weight_list):
        self.W0 = torch.tensor(weight_list[0], dtype=DTYPE, device=DEVICE)
        self.W1 = torch.tensor(weight_list[1], dtype=DTYPE, device=DEVICE)
        self.W2 = torch.tensor(weight_list[2], dtype=DTYPE, device=DEVICE)

    def __call__(self, input_data):
        # input_data = torch.tensor(input_data[:,None,:], dtype=DTYPE, device=DEVICE)
        # potentials = torch.relu(torch.tanh(torch.matmul(input_data, self.W0)))
        # potentials = torch.relu(torch.tanh(torch.matmul(potentials, self.W1)))
        # output = torch.relu(torch.tanh(torch.matmul(potentials, self.W2)))
        # return output.reshape((output.shape[0], output.shape[2])).cpu().numpy()
        
        input_data = torch.tensor(input_data, dtype=DTYPE, device=DEVICE)
        potentials = torch.relu(torch.tanh(torch.einsum("ab, abd -> ad", input_data, self.W0)))
        potentials = torch.relu(torch.tanh(torch.einsum("ab, abd -> ad", potentials, self.W1)))
        output = torch.relu(torch.tanh(torch.einsum("ab, abd -> ad", potentials, self.W2)))
        return output.cpu().numpy()