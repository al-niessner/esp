'''test the code without a pipeline'''

import dawgie.db
import dawgie.security
import dawgie.util
import excalibur.runtime.algorithms
import excalibur.runtime.bot
import os

if 'EXCALIBUR_PRIVATE_PIPELINE_INDEPENDENT' in os.environ:
    # need to fake some dawgie stuff so that can run the unit independent of any
    # pipeline since it is the data in the configuration file that matters not any
    # data or state of the pipeline itself. There going to
    # pylint: disable=missing-class-docstring,missing-function-docstring
    class FakeDawgie:
        def __init__(self):
            self.__name = 'runtime'
        def _bot(self): return self
        def _runid(self): return self
        def name(self): return self.__name
        def ds(self): return self
        def update(self): return self

    def connect(_alg, _bit, _tn):
        return FakeDawgie()

    def targets():
        return ['a','b','c', 'GJ 3193', 'GJ 3193 (taurex sim @TS)']
    # pylint: enable=missing-class-docstring,missing-function-docstring
    setattr (dawgie.db, 'connect', connect)
    setattr (dawgie.db, 'targets', targets)
    test = excalibur.runtime.algorithms.create()
    test.run (FakeDawgie())
else:
    fep = os.environ.get('FE_PORT', None)
    rid = int(os.environ.get('RUNID', None))
    tn = os.environ.get('TARGET_NAME', None)

    if fep: dawgie.util.set_ports(int(fep))

    dawgie.security.initialize(os.path.expandvars
                               (os.path.expanduser
                                (dawgie.context.guest_public_keys)))
    dawgie.db.reopen()

    if tn in ['', '__all__']:
        name = 'create'
        subtasks = excalibur.runtime.bot.AnalysisTeam('runtime', 4, rid)
        pass
    else:
        name = 'autofill'
        subtasks = excalibur.runtime.bot.TaskTeam('runtime', 4, rid, tn)
        pass
    subtasks.do(name)
    dawgie.db.close()
    dawgie.security.finalize()
    pass
