'''cerberus bounds ds'''
# -- IMPORTS --------------------------------------------------------
# import dawgie
# import excalibur
import numpy as np
import logging; log = logging.getLogger(__name__)
# -------------------------------------------------------------------
def setPriorBound():
    '''
    Set prior constraints on the spectrum-fitting parameters
    '''

    prior_ranges = {}

    prior_ranges['Tfactor'] = (0.75, 1.5)  # multiply this times T_eq

    prior_ranges['dexRange'] = (-6,6)      # use this for [X/H],[C/O],[N/O]

    prior_ranges['CTP'] = (-6,1)
    prior_ranges['HScale'] = (-6,6)
    prior_ranges['HLoc'] = (-6,1)
    prior_ranges['HThick'] = (1,20)
    prior_ranges['HIndex'] = (-4,0)        # not used; older param for when sphshell is false

    return prior_ranges

def getProfileLimits_HSTG141():
    '''
    Define limits on spectrum-fitting parameters on a target-by-target basis
    '''

    limits = {}
    limits['55 Cnc e'] = [['T',2500,'<']]
    # limits['GJ 1132 b'] = [['T',0,'>']]
    # limits['GJ 1214 b'] = [['T',1000,'<']]
    limits['GJ 3470 b'] = [['T',800,'<']]
    # limits['GJ 436 b'] = [['T',0,'>']]
    limits['GJ 9827 d'] = [['T',1000,'<']]
    # limits['HAT-P-1 b'] = [['T',2000,'<']]
    limits['HAT-P-3 b'] = [['T',1500,'<']]
    # limits['HAT-P-11 b'] = [['T',1500,'<'],  # no effect
    limits['HAT-P-11 b'] = [['HScale',-2,'<']]
    # limits['HAT-P-12 b'] = [['T',0,'>']]  # no effect
    # limits['HAT-P-17 b'] = [['T',1500,'<'],  # no effect
    limits['HAT-P-17 b'] = [['CH4',-1,'>'],  # its flat
                            ['H2CO',0,'<'],
                            ['HScale',0,'<']]
    # limits['HAT-P-18 b'] = [['T',1500,'<']]  # no effect
    # limits['HAT-P-26 b'] = [['T',1500,'<']]  # no effect.   nice spectrum
    limits['HAT-P-32 b'] = [['T',2000,'<']]
    limits['HAT-P-38 b'] = [['T',1500,'<']]
    limits['HAT-P-41 b'] = [['T',2500,'<']]
    limits['HD 97658 b'] = [['T',1000,'<']]
    limits['HD 149026 b'] = [['T',2000,'<'],
                             ['C2H2',0,'>']]
    # limits['HD 189733 b'] = [['T',0,'>']]
    # limits['HD 209458 b'] = [['T',1000,'>']]  # nice spectrum,  no effect. not in notebook list!
    # limits['K2-3 c'] = [['T',700,'<']]
    # large difference in results!
    # limits['K2-18 b'] = [['T',600,'<'],  # no effect
    #                     ['TEC[0]',0,'>']]  # undefined. modify if needed
    limits['KELT-11 b'] = [['T',2500,'<'],
                           ['C2H2',-2,'<']]
    # limits['TRAPPIST-1 b'] = [['T',600,'<']]
    # limits['TRAPPIST-1 c'] = [['T',600,'<']]
    limits['WASP-6 b'] = [['T',1500,'<']]
    limits['WASP-12 b'] = [['T',2000,'>']]
    limits['WASP-17 b'] = [['T',2000,'<']]
    # limits['WASP-29 b'] = [['T',1500,'<']]
    limits['WASP-31 b'] = [['T',2000,'<']]
    # limits['WASP-39 b'] = [['T',0,'>']]
    limits['WASP-43 b'] = [['T',1750,'<']]
    # large difference in results!
    limits['WASP-52 b'] = [['T',1000,'>'],  # slight effect at edge.  but it's flat anyway
                           # ['T',2000,'<'],  # no effect
                           ['CH4',0,'>'],
                           ['HScale',0,'<']]  # has a jump suggesting should be >0 maybe?
    limits['WASP-63 b'] = [['T',2000,'<']]
    # limits['WASP-69 b'] = [['T',2000,'<'],  # no effect
    limits['WASP-69 b'] = [['HThick',5,'<'],
                           ['HScale',-0.4,'<']]
    limits['WASP-74 b'] = [['T',2000,'<']]  # very interesting T cutoff for DISEQ. why? and CH4.  doesn't seem like the T profiling is really necessary though
    limits['WASP-76 b'] = [['T',2500,'<']]
    limits['WASP-79 b'] = [['T',2500,'<']]
    # limits['WASP-80 b'] = [['T',1500,'<']]
    # limits['WASP-107 b'] = [['T',1000,'<']]
    limits['WASP-121 b'] = [['T',2600,'<']]
    limits['XO-1 b'] = [['T',1500,'<']]
    limits['XO-2 b'] = [['T',1750,'<']]

    return limits

def applyProfiling(target, limits, alltraces, allkeys):
    '''
    Cull the spectrum-fit walkers on a target-by-target basis
    (returns proftrace, a boolean array indicating which walkers should be kept)
    '''

    appliedLimits = []
    proftrace = np.ones(len(alltraces[0]), dtype=int)
    if target in limits:
        for limit in limits[target]:
            if limit[0] in allkeys:
                log.warning('--< Found a profiling limit for: %s %s >--',target,limit)

                appliedLimits.append(limit)

                if limit[2]=='>':
                    proftrace[np.where(alltraces[allkeys.index(limit[0])] <= limit[1])] = 0
                    if len(np.where(alltraces[allkeys.index(limit[0])] <= limit[1])[0])==0:
                        log.warning('--< Profiling has no effect: %s %s >--',target,limit)
                else:
                    proftrace[np.where(alltraces[allkeys.index(limit[0])] >= limit[1])] = 0
                    if len(np.where(alltraces[allkeys.index(limit[0])] >= limit[1])[0])==0:
                        log.warning('--< Profiling has no effect: %s %s >--',target,limit)

    return proftrace, appliedLimits
