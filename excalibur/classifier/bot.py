import dawgie

import excalibur.classifier.algorithms as clsalg

class Actor(dawgie.Task):
    def list(self)->[dawgie.Task]:
        return [
            clsalg.inference()
        ]
    pass
