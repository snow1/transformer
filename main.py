from dtu import Parameters, dtu
from Trans import main as transformer
from LSTM import main as lstm
from linear import main as linear
@dtu
class Defaults(Parameters):
    name: str = "local"
    instances: int = 1
    GPU: bool = False
    time: int = 360000

    model: str = "lstm"

    def run(self, isServer: bool, model: str) -> None:
        if model == "transformer":
            transformer()
        elif model == "lstm":
            lstm()
        elif model == "linear":
            linear()


Defaults.start()
