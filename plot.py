import pandas as pd
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    for csv_file in os.listdir("best_model_ppo"):
        if csv_file.endswith(".csv") and csv_file != "win_rates.csv":
            reward_log = pd.read_csv(os.path.join("best_model_ppo", csv_file), index_col='timesteps')
            plot = reward_log.plot()
            fig = plot.get_figure()
            fig.savefig(os.path.join("best_model_ppo", csv_file.replace(".csv", ".png")))