import torch
import torch.nn as nn
import torch.optim as optim

class SWSM:
    def __init__(self, model: nn.Sequential, batch_size: int):
        self.model = model
        self.batch_size = batch_size
        self.buffer = []
        self.criterion = nn.MSELoss()  # Replace with the appropriate loss function
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)  # Replace with chosen optimizer

    def accumulate_and_step(self, input_tensor: torch.Tensor, training: bool, target_tensor: torch.Tensor = None) -> torch.Tensor:
        # Add the current sample to the buffer
        self.buffer.append((input_tensor, target_tensor))

        # Maintain sliding window by removing oldest sample if buffer exceeds batch_size
        if len(self.buffer) > self.batch_size:
            self.buffer.pop(0)

        # Check if buffer is full (for training/inference)
        if training and len(self.buffer) < self.batch_size:
            return None, None  # Not enough samples yet, wait to fill the buffer

        # Process the batch depending on training mode
        batch_inputs, batch_targets = self.prepare_batch(training)
        if training:
            return None, self.train_step(batch_inputs, batch_targets)
        else:
            return self.infer_step(input_tensor), None

    def prepare_batch(self, training: bool):
        """Prepare batch inputs and targets from the buffer."""
        batch_inputs = torch.stack([sample[0] for sample in self.buffer])
        batch_targets = torch.stack([sample[1] for sample in self.buffer]) if training else None
        return batch_inputs, batch_targets

    def train_step(self, batch_inputs: torch.Tensor, batch_targets: torch.Tensor):
        """Perform a training step on the batch."""
        self.model.train()
        if batch_targets is None:
            raise ValueError("Target tensor must be provided for training.")

        # Zero the gradients
        self.optimizer.zero_grad()

        # Forward pass
        output = self.model(batch_inputs)

        # Calculate loss
        loss = self.criterion(output, batch_targets)

        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()

        return loss

    def infer_step(self, batch_inputs: torch.Tensor):
        """Perform an inference step on the batch."""
        self.model.eval()
        with torch.no_grad():
            output = self.model(batch_inputs)
        return output
    
'''
if __name__ == "__main__":
    model = nn.Sequential(
        nn.Linear(3, 10),
        nn.Sigmoid(),
        nn.Linear(10,10),
        nn.Sigmoid(),
        nn.Linear(10,1)
    )
    swsm = SWSM(model, 10)
    bo = True
    for i in range(0,1000):
        input = torch.Tensor([1, 2, 2])
        target = torch.Tensor([15])
        a,b = swsm.accumulate_and_step(input, bo, target)
        print("A: " + str(a))
        print("B: " + str(b))
        if (b is not None and b.item() < 2):
            bo = False

'''
