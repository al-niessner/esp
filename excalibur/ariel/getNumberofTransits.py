''' module to read in the number of transits for a given target '''

import csv

def readArielTargetList(dataDir='/proj/data/ariel/',
                        filename='Known_Tier1_20220525.csv'):
    '''
    Read in the Edwards list of Ariel targets.
    (contains a column for the #-of-planned-transit-observations)
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

    ariel_target_info = readArielTargetList()

    observing_plan = {}

    for target in ariel_target_info:
        observing_plan[target['Planet Name']] = target['Tier 1 Observations']

    return observing_plan
