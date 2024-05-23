import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot_score(scores, filename="progress_score.jpg", title="Game Scores Progress"):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title(title)
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, color='blue', marker='o', linestyle='None')
    plt.ylim(ymin=-10)
    plt.ylim(ymax=10)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.savefig(filename)

def plot_won(games_won, filename="progress_wins.jpg", title="Game Won Progress"):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title(title)
    plt.xlabel('Number of Games')
    plt.ylabel('Games Won')
    plt.plot(games_won, color='red')
    plt.ylim(ymin=0)
    plt.text(len(games_won)-1, games_won[-1], str(games_won[-1]))
    plt.savefig(filename)
