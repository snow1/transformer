from dtu import Parameters, dtu
from Trans import main as transformer
from Trans2 import main as transformer2
from LSTM import main as lstm
from linear import main as linear
from model_CWGAN import main as CWGAN
from model_ACGAN import main as ACGAN
from CNN import main as cnn
from main_Transformer import main as maintransformer

@dtu
class Defaults(Parameters):
    name: str = "local"
    instances: int = 1
    GPU: bool = False
    time: int = 360000

    model: str = "cnn"

    def run(self, isServer: bool, model: str) -> None:
        if model == "transformer":
            transformer()
        elif model == "lstm":
            lstm()
        elif model == "linear":
            linear()
        elif model == "transformer2":
            transformer2()
        elif model == "CWGAN":
            CWGAN()
        elif model == "ACGAN":
            ACGAN()
        elif model == "cnn":
            cnn()
        elif model == "maintransformer":
            maintransformer()


Defaults.start()
