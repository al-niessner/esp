''' module to read in the number of transits for a given target '''
import logging; log = logging.getLogger(__name__)
import csv

def readArielTargetList(dataDir='/proj/data/ariel/',
                        filename='AR_MRS_12022019.csv'):
    '''
    Generic routine to read in an Ariel target list.
    Default is the 2019 spreadsheet that says which Tier each target is in.
    (The 2022 spreadsheet from Billy doesn't have this)
    '''

    with open(dataDir+filename,'r',encoding='ascii') as file:
        csvFile = csv.DictReader(file)

        listofDictionaries = []
        # print('starting to read',filename)
        for line in csvFile:
            listofDictionaries.append(line)

    return listofDictionaries

def read_numberofTransits_table_Edwards22():
    '''
    Extract #-of-planned-transit-observations for each target
    based on the Edwards 2022 list of Ariel targets
    '''

    ariel_target_info = readArielTargetList(filename='Known_Tier1_20220525.csv')

    observing_plan = {}

    for target in ariel_target_info:
        observing_plan[target['Planet Name']] = target['Tier 1 Observations']

    return observing_plan

def read_numberofTransits_table_MCS(includeEclipseTargets=False):
    '''
    Extract #-of-planned-transit-observations for each target.
    based on the Ariel Mission Candidate Sample (MCS)
    https://github.com/arielmission-space/Mission_Candidate_Sample
    '''

    # only include known planets (which all have planet masses)
    ariel_target_info = readArielTargetList(filename='Ariel_MCS_Known_2024-02-14.csv')

    observing_plan = {}
    for target in ariel_target_info:
        # only include transit targets, not eclipse targets
        if target['Preferred Method']=='Transit' or target['Preferred Method']=='Either':

            if target['Tier 2 Observations']!=target['Tier 2 Transits']:
                # print('tier 2 obs,transits',target['Planet Name'],
                #       target['Tier 2 Observations'],target['Tier 2 Transits'])
                log.warning('--< ERROR: #-observations should equal #-transits: %s >--',
                            target['Planet Name'])
                # exit('ERROR: tier 2 observations should equal the number of transits')

            # print('# of visits',target['Planet Name'],target['Tier 2 Observations'])
            observing_plan[target['Planet Name']] = {
                'tier':2,
                'number of visits':target['Tier 2 Transits']}

        elif target['Preferred Method']=='Eclipse':
            if includeEclipseTargets:
                observing_plan[target['Planet Name']] = {
                    'tier':2,
                    'number of visits':target['Tier 2 Transits']}
            else:
                # print('skipping an eclipse target',target['Planet Name'])
                pass
        else:
            log.warning('--< ERROR: unknown observing method: %s %s >--',
                        target['Preferred Method'],target['Planet Name'])
            pass

    return observing_plan

def make_tier_table():
    '''
    Extract the Tier level for each target and then the number of visits, based on that Tier.
    (Almost the same as the previous routine, but that one assumed Tier 1.)
    '''

    observing_plan = {}

    # start with the older 2019 list that specifies tier level and # of visits
    edwards2019_target_info = readArielTargetList()

    for target in edwards2019_target_info:
        name = target['Planet Name']
        observing_plan[name] = {}

        tierInteger = f"{float(target['Tier']):.0f}"
        observing_plan[name]['tier'] = tierInteger

        # set the number of observations based on the Tier level
        observing_plan[name]['number of visits'] = \
            f"{float(target['Tier '+tierInteger+' Observations']):.0f}"

    # overwrite info from the newer 2024 list
    MCS_target_info = read_numberofTransits_table_MCS(includeEclipseTargets=True)
    # for name in MCS_target_info:
    #    targetInfo = MCS_target_info[name]
    for name, targetInfo in MCS_target_info.items():
        # print('check below',name)
        # print('check below',MCS_target_info[name])
        # print('check below',targetInfo)

        if name in observing_plan:
            # print('this target is in both lists:',name)
            # print('    tier change?',observing_plan[name]['tier'],targetInfo['tier'])
            # print('   visit change?',observing_plan[name]['number of visits'],
            #                          targetInfo['number of visits'])
            pass
        else:
            observing_plan[name] = {}
            # print('this is a new target, not in previous Ariel list:',name)

        # set Tier level (should always be 2)
        observing_plan[name]['tier'] = f"{float(targetInfo['tier']):.0f}"
        if observing_plan[name]['tier']!='2':
            print('STRANGE: should only be considering tier 2')

        # set the number of observations based on the Tier level
        observing_plan[name]['number of visits'] = f"{float(targetInfo['number of visits']):.0f}"

    return observing_plan
