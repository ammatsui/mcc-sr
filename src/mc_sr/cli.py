import argparse
import numpy as np

def calc_mse(x, y):
    a, b, c = 1.0, 1.0, 1.7
    y_pred = a * x ** 2 + b * np.sin(c * x)
    return np.mean((y - y_pred) ** 2)

def run_mwe(seed=123):
    np.random.seed(seed)
    x = np.linspace(-2, 2, 512)
    y = np.sin(1.7 * x) + x ** 2 + np.random.normal(0, 0.02, x.shape)
    mse = calc_mse(x, y)
    print(f"MWE: Toy equation MSE = {mse:.6f}")

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    mwe_parser = subparsers.add_parser("mwe", help="Run minimal working example")
    args = parser.parse_args()

    if args.command == "mwe":
        run_mwe()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()