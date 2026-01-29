import argparse
import platform
import sys
import time

import torch
import torch.nn as nn

class TinyCNN(nn.Module):
    """Basic CNN used only for environment validation."""
    def __init__(self, in_channels: int = 1, num_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def main():
    parser = argparse.ArgumentParser(description="CNN validation test (CPU/GPU)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    print("Basic CNN Validation Test")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA runtime: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TinyCNN(
        in_channels=args.channels,
        num_classes=args.classes,
    ).to(device)

    x = torch.randn(
        args.batch_size,
        args.channels,
        args.image_size,
        args.image_size,
        device=device,
    )

    model.eval()
    with torch.no_grad():
        start = time.time()
        for _ in range(args.steps):
            y = model(x)
        end = time.time()

    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
    print(f"Avg forward time: {(end - start) / args.steps:.6f} sec")

    # Backward pass check
    model.train()
    x2 = torch.randn_like(x)
    y2 = model(x2)
    loss = y2.mean()
    loss.backward()
    print("Backward pass: OK")
    print("Validation test completed successfully")


if __name__ == "__main__":
    main()