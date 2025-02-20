'''monkey patch some modules'''

import ldtk


class LDPSet(ldtk.LDPSet):
    '''
    A. NIESSNER: INLINE HACK TO ldtk.LDPSet
    '''

    @staticmethod
    def is_mime():
        '''is_mime ds'''
        return True

    @property
    def profile_mu(self):
        '''profile_mu ds'''
        return self._mu

    pass


setattr(ldtk, 'LDPSet', LDPSet)
setattr(ldtk.ldtk, 'LDPSet', LDPSet)
