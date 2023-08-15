''' module to read in the number of transits for a given target '''

import csv

def readArielTargetList(dataDir='/proj/data/ariel/',
                        filename='AR_MRS_12022019.csv'):
    '''
    Read in the 2019 spreadsheet that says which Tier each target is in.
    (The 2022 spreadsheet from Billy doesn't have this)
    '''

    with open(dataDir+filename,'r',encoding='ascii') as file:
        csvFile = csv.DictReader(file)

        listofDictionaries = []
        # print('starting to read',filename)
        for line in csvFile:
            listofDictionaries.append(line)

    return listofDictionaries

def make_numberofTransits_table():
    '''
    Extract #-of-planned-transit-observations for each target.
    (based on the Edwards list of Ariel targets)
    '''

    ariel_target_info = readArielTargetList(filename='Known_Tier1_20220525.csv')

    observing_plan = {}

    for target in ariel_target_info:
        observing_plan[target['Planet Name']] = target['Tier 1 Observations']

    return observing_plan

def make_tier_table():
    '''
    Extract the Tier level for each target and then the number of visits, based on that Tier.
    (Almost the same as the previous routine, but that one assumed Tier 1.)
    '''

    ariel_target_info = readArielTargetList()

    observing_plan = {}

    for target in ariel_target_info:
        name = target['Planet Name']
        observing_plan[name] = {}

        tierInteger = f"{float(target['Tier']):.0f}"
        observing_plan[name]['tier'] = tierInteger

        # set the number of observations based on the Tier level
        observing_plan[name]['number of visits'] = \
            f"{float(target['Tier '+tierInteger+' Observations']):.0f}"

    return observing_plan
