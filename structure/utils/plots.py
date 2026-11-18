import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_durations(agent, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(agent.episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)

    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def get_moving_avgs(arr, window, convolution_mode):
    return np.convolve(np.array(arr).flatten(), np.ones(window), mode=convolution_mode) / window


def plot_agent(agent, rolling_length=500):
    fig, axs = plt.subplots(ncols=3, figsize=(15, 5))

    axs[0].set_title("Win rate")
    reward_moving_average = get_moving_avgs(agent.wins_rates, rolling_length, "valid")
    axs[0].plot(range(len(agent.wins_rates)), agent.wins_rates)
    axs[0].set_ylabel("Média de reward (~ taxa de vitória)")
    axs[0].set_xlabel("Episódio")

    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(agent.episode_durations, rolling_length, "valid")
    axs[1].plot(range(len(agent.episode_durations)), agent.episode_durations)
    axs[1].set_ylabel("Comprimento médio do episódio")
    axs[1].set_xlabel("Episódio")

    axs[2].set_title("Training Error")
    training_error_moving_average = get_moving_avgs(agent.losses, rolling_length, "same")
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    axs[2].set_ylabel("TD Error")
    axs[2].set_xlabel("Step")

    plt.tight_layout()
    plt.show()


def plot_epsilon_decay(agent):
    """Mostra como o epsilon (exploração) caiu ao longo dos episódios."""
    plt.figure(figsize=(8, 4))
    plt.plot(agent.eps_history)
    plt.title("Decaimento do epsilon (exploração vs exploração)")
    plt.xlabel("Step")
    plt.ylabel("Epsilon")
    plt.axhline(agent.eps_history[-1], color="red", linestyle="--", label="epsilon mínimo")
    plt.legend()
    plt.tight_layout()
    plt.show()