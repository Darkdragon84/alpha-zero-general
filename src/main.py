import logging

from coach import Coach
# from othello.OthelloGame import OthelloGame as Game
# from othello.keras.NNet import NNetWrapper
from tictactoe.TicTacToeGame import TicTacToeGame as Game
from tictactoe.keras.NNet import NNetWrapper
from utils import dotdict

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

args = dotdict({
    'numIters': 10,
    'numEps': 100,  # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,  #
    'updateThreshold': 0.6,
    # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,  # Number of game examples to train the neural networks.
    'numMCTSSims': 10,  # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,
    # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})


def main():
    log.info('Loading %s...', Game.__name__)
    game = Game(6)

    log.info('Loading %s...', NNetWrapper.__name__)
    nnet = NNetWrapper(game)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0],
                 args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    coach = Coach(game, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        coach.load_train_examples()

    log.info('Starting the learning process 🎉')
    coach.learn()


if __name__ == "__main__":
    main()
