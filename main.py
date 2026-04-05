import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# Social Math Neural Network
# -----------------------------
# Each layer corresponds to a conceptual stage:
# Identity → Direction → Relationships → Influence → Forces → Balance
# → Sustainability → Distortion → Recalibration → Legacy

class SocialMathNet(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=32):
        super(SocialMathNet, self).__init__()

        # POINT → Identity
        self.identity = nn.Linear(input_dim, hidden_dim)

        # LINE → Direction
        self.direction = nn.Linear(hidden_dim, hidden_dim)

        # PLANE → Relationships
        self.relationships = nn.Linear(hidden_dim, hidden_dim)

        # VOLUME → Influence
        self.influence = nn.Linear(hidden_dim, hidden_dim)

        # VECTORS → Forces
        self.forces = nn.Linear(hidden_dim, hidden_dim)

        # AXES → Balance
        self.balance = nn.Linear(hidden_dim, hidden_dim)

        # SYMMETRY → Justice
        self.symmetry = nn.Linear(hidden_dim, hidden_dim)

        # EQUATION → Sustainability
        self.sustainability = nn.Linear(hidden_dim, hidden_dim)

        # DISTORTION → Collapse Risk
        self.distortion = nn.Linear(hidden_dim, hidden_dim)

        # RECALIBRATION → Correction
        self.recalibration = nn.Linear(hidden_dim, hidden_dim)

        # LEGACY → Final Output
        self.legacy = nn.Linear(hidden_dim, 1)

        self.activation = nn.ReLU()

    def forward(self, x):
        stages = {}

        x = self.activation(self.identity(x))
        stages['identity'] = x

        x = self.activation(self.direction(x))
        stages['direction'] = x

        x = self.activation(self.relationships(x))
        stages['relationships'] = x

        x = self.activation(self.influence(x))
        stages['influence'] = x

        x = self.activation(self.forces(x))
        stages['forces'] = x

        x = self.activation(self.balance(x))
        stages['balance'] = x

        x = self.activation(self.symmetry(x))
        stages['symmetry'] = x

        x = self.activation(self.sustainability(x))
        stages['sustainability'] = x

        x = self.activation(self.distortion(x))
        stages['distortion'] = x

        x = self.activation(self.recalibration(x))
        stages['recalibration'] = x

        output = self.legacy(x)
        stages['legacy'] = output

        return output, stages


# -----------------------------
# Synthetic Training Data
# -----------------------------
def generate_data(num_samples=1000, input_dim=16):
    X = torch.randn(num_samples, input_dim)

    # Social stability score (synthetic target)
    # Simulates "balanced system" objective
    y = (
        X.mean(dim=1)
        + 0.5 * torch.sin(X.sum(dim=1))
        - 0.3 * torch.var(X, dim=1)
    ).unsqueeze(1)

    return X, y


# -----------------------------
# Training Loop
# -----------------------------
def train():
    model = SocialMathNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    X, y = generate_data()

    for epoch in range(200):
        optimizer.zero_grad()

        output, _ = model(X)
        loss = criterion(output, y)

        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch} | Loss {loss.item():.4f}")

    return model


# -----------------------------
# Inference Demo
# -----------------------------
def demo(model):
    test_input = torch.randn(1, 16)
    output, stages = model(test_input)

    print("\n--- Social Math Inference ---")
    print(f"Final Legacy Output: {output.item():.4f}")

    for key, value in stages.items():
        print(f"{key.upper()} mean activation: {value.mean().item():.4f}")


if __name__ == "__main__":
    model = train()
    demo(model)
