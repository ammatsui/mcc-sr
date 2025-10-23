import pandas as pd
import matplotlib.pyplot as plt


def plot_mse_trends(csv_path, save_path=None):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10,6))
    plt.scatter(df['generation'], df['eq_mse_min'], label='Min Eq MSE')
    plt.scatter(df['generation'], df['eq_mse_median'], label='Median Eq MSE')
    plt.scatter(df['generation'], df['eq_mse_max'], label='Max Eq MSE')
    plt.xlabel('Generation')
    plt.ylabel('Equation MSE')
    plt.title('Equation MSE Evolution')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_num_passed(csv_path, save_path=None):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8,4))
    plt.scatter(df['generation'], df['num_eq_passed'], label='Eq Passed MC')
    plt.scatter(df['generation'], df['num_gen_passed'], label='Gen Passed MC')
    plt.xlabel('Generation')
    plt.ylabel('Number Passed')
    plt.title('MC Gate Passing Over Time')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_thresholds(csv_path, save_path=None):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8,4))
    plt.scatter(df['generation'], df['tau'], label='tau')
    plt.scatter(df['generation'], df['tau_prime'], label='tau_prime')
    plt.xlabel('Generation')
    plt.ylabel('Threshold Value')
    plt.title('MC Thresholds Over Generations')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_tree_sizes(csv_path, save_path=None):
    """
    Plot min, mean, and max tree sizes over generations.
    """
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10,6))
    plt.scatter(df['generation'], df['eq_tree_size_min'], label='Min Tree Size')
    plt.scatter(df['generation'], df['eq_tree_size_mean'], label='Mean Tree Size')
    plt.scatter(df['generation'], df['eq_tree_size_max'], label='Max Tree Size')
    plt.xlabel('Generation')
    plt.ylabel('Equation Tree Size')
    plt.title('Equation Tree Size Evolution')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_metrics(csv_path, out_dir=None):
    plot_mse_trends(csv_path, f"{out_dir}/mse_trends.png" if out_dir else None)
    plot_num_passed(csv_path, f"{out_dir}/num_passed.png" if out_dir else None)
    plot_thresholds(csv_path, f"{out_dir}/thresholds.png" if out_dir else None)
    plot_tree_sizes(csv_path, f"{out_dir}/tree_sizes.png" if out_dir else None)
    print("All plots generated.")