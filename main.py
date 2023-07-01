from dtu import Parameters, dtu
from Trans import main as transformer
from Trans2 import main as transformer2
from Trans3 import main as transformer3
from Trans4 import main as transformer4
from Trans5 import main as transformer5
from Trans6 import main as transformer6
from LSTM import main as lstm
from linear import main as linear
from model_CWGAN import main as CWGAN
from model_ACGAN import main as ACGAN
from CNN import main as cnn

#from mainTransformer import main as maintransformer


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
        elif model == "transformer3":
            transformer3()
        elif model == "transformer4":
            transformer4()
        elif model == "transformer5":
            transformer5()
        elif model == "transformer6":
            transformer6()


Defaults.start()
