from dtu import Parameters, dtu
from Trans import main

@dtu
class Defaults(Parameters):
    name: str = "local"
    instances: int = 1
    GPU: bool = False
    time: int = 360000

    b: float = 2.0
    a: int = 1
    d: str = "fd"

    def run(self, isServer: bool) -> None:
        main()
        #print(b, d, self.time, isServer)


Defaults.start()
