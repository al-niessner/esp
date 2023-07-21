'''target targetlists ds'''

# --------------------------------------------------------------------
def targetlist_active():
    '''
    all good targets (everything except targetlist_junk)
    '''
    targets = [
        '55 Cnc',
        'AU Mic',
        'CoRoT-1',
        'CoRoT-11',
        'CoRoT-19',
        'CoRoT-2',
        'CoRoT-5',
        'CoRoT-7',
        'DS Tuc A',
        'EPIC 211945201',
        'EPIC 246851721',
        'G 9-40',
        'GJ 1132',
        'GJ 1214',
        'GJ 1252',
        'GJ 3053',
        # 'GJ 3193',
        'GJ 3470',
        'GJ 3473',
        'GJ 357',
        'GJ 367',
        'GJ 3929',
        'GJ 436',
        'GJ 486',
        'GJ 9827',
        'GPX-1',
        'HAT-P-1',
        'HAT-P-11',
        'HAT-P-12',
        'HAT-P-13',
        'HAT-P-14',
        'HAT-P-15',
        'HAT-P-16',
        'HAT-P-17',
        'HAT-P-18',
        'HAT-P-19',
        'HAT-P-2',
        'HAT-P-20',
        'HAT-P-21',
        'HAT-P-22',
        'HAT-P-23',
        'HAT-P-24',
        'HAT-P-25',
        'HAT-P-26',
        'HAT-P-27',
        'HAT-P-28',
        'HAT-P-29',
        'HAT-P-3',
        'HAT-P-30',
        'HAT-P-31',
        'HAT-P-32',
        'HAT-P-33',
        'HAT-P-34',
        'HAT-P-35',
        'HAT-P-36',
        'HAT-P-37',
        'HAT-P-38',
        'HAT-P-39',
        'HAT-P-4',
        'HAT-P-40',
        'HAT-P-41',
        'HAT-P-42',
        'HAT-P-43',
        'HAT-P-44',
        'HAT-P-45',
        'HAT-P-46',
        'HAT-P-49',
        'HAT-P-5',
        'HAT-P-50',
        'HAT-P-51',
        'HAT-P-52',
        'HAT-P-53',
        'HAT-P-54',
        'HAT-P-55',
        'HAT-P-56',
        'HAT-P-57',
        'HAT-P-58',
        'HAT-P-59',
        'HAT-P-6',
        'HAT-P-60',
        'HAT-P-61',
        'HAT-P-62',
        'HAT-P-64',
        'HAT-P-65',
        'HAT-P-66',
        'HAT-P-67',
        'HAT-P-68',
        'HAT-P-69',
        'HAT-P-7',
        'HAT-P-70',
        'HAT-P-8',
        'HAT-P-9',
        'HATS-1',
        'HATS-11',
        'HATS-13',
        'HATS-18',
        'HATS-2',
        'HATS-23',
        'HATS-24',
        'HATS-25',
        'HATS-26',
        'HATS-27',
        'HATS-28',
        'HATS-29',
        'HATS-3',
        'HATS-30',
        'HATS-31',
        'HATS-33',
        'HATS-34',
        'HATS-35',
        'HATS-37 A',
        'HATS-38',
        'HATS-39',
        'HATS-4',
        'HATS-40',
        'HATS-41',
        'HATS-42',
        'HATS-43',
        'HATS-46',
        'HATS-47',
        'HATS-48 A',
        'HATS-5',
        'HATS-50',
        'HATS-51',
        'HATS-52',
        'HATS-53',
        'HATS-56',
        'HATS-57',
        'HATS-58 A',
        'HATS-6',
        'HATS-60',
        'HATS-62',
        'HATS-64',
        'HATS-65',
        'HATS-67',
        'HATS-68',
        'HATS-7',
        'HATS-70',
        'HATS-72',
        'HATS-9',
        'HD 106315',
        'HD 108236',
        'HD 110082',
        'HD 110113',
        'HD 136352',
        'HD 1397',
        'HD 149026',
        'HD 152843',
        'HD 15337',
        'HD 17156',
        'HD 183579',
        # 'HD 185603',
        'HD 189733',
        'HD 191939',
        # 'HD 195689',
        # 'HD 197481',
        'HD 202772 A',
        'HD 207897',
        'HD 209458',
        'HD 213885',
        'HD 219134',
        'HD 219666',
        'HD 221416',
        'HD 23472',
        'HD 2685',
        'HD 332231',
        'HD 5278',
        'HD 63433',
        'HD 63935',
        'HD 73583',
        'HD 86226',
        'HD 89345',
        'HD 97658',
        # 'HIP 41378',
        'HIP 65 A',
        'HIP 67522',
        'HR 858',
        'K2-107',
        'K2-116',
        'K2-121',
        'K2-124',
        'K2-129',
        'K2-132',
        'K2-136',
        'K2-138',
        'K2-139',
        'K2-140',
        'K2-141',
        'K2-155',
        'K2-167',
        'K2-174',
        'K2-18',
        'K2-19',
        'K2-198',
        'K2-21',
        'K2-212',
        'K2-22',
        'K2-222',
        'K2-232',
        'K2-237',
        'K2-238',
        'K2-239',
        'K2-24',
        'K2-25',
        'K2-26',
        'K2-260',
        'K2-261',
        'K2-266',
        'K2-28',
        'K2-280',
        'K2-284',
        'K2-287',
        'K2-289',
        'K2-29',
        'K2-295',
        'K2-3',
        'K2-31',
        'K2-32',
        'K2-329',
        'K2-33',
        'K2-333',
        'K2-334',
        'K2-34',
        'K2-353',
        'K2-36',
        'K2-39',
        'K2-403',
        'K2-405',
        'K2-406',
        'K2-52',
        'K2-53',
        'K2-55',
        'K2-58',
        'K2-79',
        'K2-87',
        'K2-9',
        'K2-90',
        'K2-93',
        'K2-95',
        'K2-96',
        'K2-97',
        'KELT-1',
        'KELT-10',
        'KELT-11',
        'KELT-12',
        'KELT-14',
        'KELT-15',
        'KELT-16',
        'KELT-17',
        'KELT-18',
        'KELT-19 A',
        'KELT-2 A',
        'KELT-20',
        'KELT-21',
        'KELT-23 A',
        'KELT-24',
        'KELT-3',
        'KELT-4 A',
        'KELT-6',
        'KELT-7',
        'KELT-8',
        'KELT-9',
        # 'KIC 12266812',
        'KOI-13',
        'KOI-94',
        'KPS-1',
        'Kepler-10',
        'Kepler-102',
        'Kepler-104',
        'Kepler-1083',
        'Kepler-12',
        'Kepler-125',
        'Kepler-126',
        'Kepler-127',
        'Kepler-1339',
        'Kepler-138',
        'Kepler-14',
        'Kepler-1485',
        'Kepler-1492',
        'Kepler-156',
        'Kepler-1568',
        'Kepler-158',
        'Kepler-16',
        'Kepler-1625',
        'Kepler-1651',
        'Kepler-167',
        'Kepler-17',
        'Kepler-18',
        'Kepler-19',
        'Kepler-105',
        'Kepler-108',
        'Kepler-1314',
        'Kepler-1513',
        'Kepler-20',
        'Kepler-205',
        'Kepler-218',
        'Kepler-236',
        'Kepler-249',
        'Kepler-25',
        'Kepler-26',
        'Kepler-293',
        'Kepler-297',
        'Kepler-309',
        'Kepler-32',
        'Kepler-37',
        'Kepler-395',
        'Kepler-33',
        'Kepler-396',
        'Kepler-45',
        'Kepler-454',
        'Kepler-48',
        'Kepler-482',
        'Kepler-49',
        'Kepler-42',
        'Kepler-435',
        'Kepler-444',
        'Kepler-447',
        'Kepler-450',
        'Kepler-468',
        'Kepler-489',
        'Kepler-5',
        'Kepler-504',
        'Kepler-505',
        'Kepler-570',
        'Kepler-582',
        'Kepler-598',
        'Kepler-6',
        'Kepler-603',
        'Kepler-61',
        'Kepler-62',
        'Kepler-68',
        'Kepler-7',
        'Kepler-705',
        'Kepler-737',
        'Kepler-769',
        'Kepler-786',
        'Kepler-76',
        'Kepler-79',
        'Kepler-9',
        'Kepler-93',
        'Kepler-94',
        'L 98-59',
        'LHS 1478',
        'LHS 1678',
        'LHS 3844',
        'LP 714-47',
        'LP 791-18',
        'LTT 1445 A',
        'LTT 3780',
        'LTT 9779',
        'MASCARA-1',
        'MASCARA-4',
        'NGTS-10',
        'NGTS-11',
        'NGTS-12',
        'NGTS-13',
        'NGTS-2',
        'NGTS-5',
        'NGTS-6',
        'NGTS-8',
        'OGLE-TR-056',
        'OGLE-TR-10',
        'Qatar-1',
        'Qatar-10',
        'Qatar-2',
        'Qatar-4',
        'Qatar-5',
        'Qatar-6',
        'Qatar-7',
        'Qatar-8',
        'Qatar-9',
        # 'TIC 184892124',
        'TIC 257060897',
        'TOI-1064',
        'TOI-1075',
        'TOI-1130',
        'TOI-1201',
        'TOI-122',
        'TOI-1227',
        'TOI-1231',
        'TOI-125',
        'TOI-1259 A',
        'TOI-1260',
        'TOI-1266',
        'TOI-1268',
        'TOI-1296',
        'TOI-1298',
        'TOI-1333',
        'TOI-1411',
        'TOI-1431',
        'TOI-1442',
        'TOI-1478',
        'TOI-150',
        'TOI-1518',
        'TOI-157',
        'TOI-1601',
        'TOI-163',
        'TOI-1670',
        'TOI-1685',
        'TOI-169',
        'TOI-1693',
        'TOI-172',
        'TOI-1728',
        # 'TOI-175',
        'TOI-1759',
        'TOI-178',
        'TOI-1789',
        'TOI-1807',
        'TOI-1842',
        'TOI-1860',
        'TOI-1899',
        # 'TOI-193',
        'TOI-201',
        'TOI-2076',
        'TOI-2109',
        'TOI-216',
        'TOI-2260',
        'TOI-2337',
        'TOI-237',
        'TOI-2411',
        'TOI-2427',
        'TOI-257',
        'TOI-2669',
        'TOI-269',
        'TOI-270',
        'TOI-3362',
        'TOI-421',
        'TOI-431',
        'TOI-4329',
        'TOI-451',
        'TOI-481',
        'TOI-500',
        'TOI-530',
        'TOI-540',
        'TOI-544',
        'TOI-559',
        'TOI-561',
        'TOI-564',
        'TOI-620',
        'TOI-628',
        'TOI-640',
        'TOI-674',
        'TOI-677',
        'TOI-700',
        'TOI-776',
        'TOI-813',
        'TOI-824',
        'TOI-833',
        'TOI-837',
        'TOI-849',
        'TOI-892',
        'TOI-905',
        'TOI-954',
        'TRAPPIST-1',
        'TrES-1',
        'TrES-2',
        'TrES-3',
        'TrES-4',
        'TrES-5',
        'V1298 Tau',
        'WASP-1',
        'WASP-10',
        'WASP-100',
        'WASP-101',
        'WASP-103',
        'WASP-104',
        'WASP-105',
        'WASP-106',
        'WASP-107',
        'WASP-11',
        'WASP-110',
        'WASP-113',
        'WASP-114',
        'WASP-117',
        'WASP-118',
        'WASP-119',
        'WASP-12',
        'WASP-120',
        'WASP-121',
        'WASP-123',
        'WASP-124',
        'WASP-126',
        'WASP-127',
        'WASP-13',
        'WASP-131',
        'WASP-132',
        'WASP-133',
        'WASP-135',
        'WASP-136',
        'WASP-138',
        'WASP-139',
        'WASP-14',
        'WASP-140',
        'WASP-141',
        'WASP-142',
        'WASP-145 A',
        'WASP-147',
        'WASP-148',
        'WASP-15',
        'WASP-151',
        'WASP-153',
        'WASP-156',
        'WASP-157',
        'WASP-158',
        'WASP-159',
        'WASP-16',
        'WASP-160 B',
        'WASP-161',
        'WASP-163',
        'WASP-164',
        'WASP-165',
        'WASP-166',
        'WASP-167',
        'WASP-168',
        'WASP-169',
        'WASP-17',
        'WASP-170',
        'WASP-172',
        'WASP-173 A',
        'WASP-174',
        'WASP-175',
        'WASP-176',
        'WASP-177',
        'WASP-178',
        'WASP-18',
        'WASP-180 A',
        'WASP-181',
        'WASP-182',
        'WASP-183',
        'WASP-184',
        'WASP-185',
        'WASP-186',
        'WASP-187',
        'WASP-189',
        'WASP-19',
        'WASP-190',
        'WASP-192',
        'WASP-2',
        'WASP-20',
        'WASP-21',
        'WASP-22',
        'WASP-23',
        'WASP-24',
        'WASP-25',
        'WASP-26',
        'WASP-28',
        'WASP-29',
        'WASP-3',
        'WASP-31',
        'WASP-32',
        'WASP-33',
        'WASP-34',
        'WASP-35',
        'WASP-36',
        'WASP-37',
        'WASP-38',
        'WASP-39',
        'WASP-4',
        'WASP-41',
        'WASP-42',
        'WASP-43',
        'WASP-44',
        'WASP-45',
        'WASP-46',
        'WASP-47',
        'WASP-48',
        'WASP-49',
        'WASP-5',
        'WASP-50',
        'WASP-52',
        'WASP-53',
        'WASP-54',
        'WASP-55',
        'WASP-56',
        'WASP-57',
        'WASP-58',
        'WASP-6',
        'WASP-61',
        'WASP-62',
        'WASP-63',
        'WASP-64',
        'WASP-65',
        'WASP-66',
        'WASP-67',
        'WASP-68',
        'WASP-69',
        'WASP-7',
        'WASP-70 A',
        'WASP-71',
        'WASP-72',
        'WASP-73',
        'WASP-74',
        'WASP-75',
        'WASP-76',
        'WASP-77',
        'WASP-78',
        'WASP-79',
        'WASP-8',
        'WASP-80',
        'WASP-81',
        'WASP-82',
        'WASP-83',
        'WASP-84',
        'WASP-85 A',
        'WASP-87',
        'WASP-88',
        'WASP-89',
        'WASP-90',
        'WASP-91',
        'WASP-92',
        'WASP-93',
        'WASP-94',
        'WASP-95',
        'WASP-96',
        'WASP-97',
        'WASP-98',
        'WASP-99',
        'Wolf 503',
        'XO-1',
        'XO-2',
        'XO-3',
        'XO-4',
        'XO-5',
        'XO-6',
        'XO-7',
        'pi Men',
    ]
    return targets
# --------------------------------------------------------------------

def targetlist_roudier62():
    '''
    The 62-planet sample from Roudier et al. 2021
    This list is actually 60 stars (2 systems have multiple planets)
    '''

    targets = [
        '55 Cnc',
        'GJ 1132',
        'GJ 1214',
        'GJ 3053',
        'GJ 3470',
        'GJ 436',
        'GJ 9827',
        'HAT-P-1',
        'HAT-P-11',
        'HAT-P-12',
        'HAT-P-17',
        'HAT-P-18',
        'HAT-P-26',
        'HAT-P-3',
        'HAT-P-32',
        'HAT-P-38',
        'HAT-P-41',
        'HAT-P-7',
        'HATS-7',
        'HD 106315',
        'HD 149026',
        'HD 189733',
        'HD 209458',
        'HD 97658',
        'K2-18',
        'K2-24',
        'K2-3',        # c and d
        'K2-33',
        'K2-93',
        'K2-96',
        'KELT-1',
        'KELT-11',
        'KELT-7',
        'Kepler-138',
        'Kepler-16',
        'TRAPPIST-1',  # b and c
        'WASP-101',
        'WASP-103',
        'WASP-107',
        'WASP-12',
        'WASP-121',
        'WASP-17',
        'WASP-18',
        'WASP-19',
        'WASP-29',
        'WASP-31',
        'WASP-39',
        'WASP-43',
        'WASP-52',
        'WASP-6',
        'WASP-62',
        'WASP-63',
        'WASP-67',
        'WASP-69',
        'WASP-74',
        'WASP-76',
        'WASP-79',
        'WASP-80',
        'XO-1',
        'XO-2',
    ]

    return targets

# --------------------------------------------------------------------

def targetlist_junk():
    '''
    targets that are just junk (mis-spellings, etc)
    '''

    targets = [
        '__all__',
        '',
        'COROT-1',    # correct spelling is CoRoT-1
        'GJ-1214',    # correct spelling is GJ 1214
        'hd 189733',  # correct spelling is HD 189733
        'HD 189733 (taurex sim @TS)',
        'HR 8799',    # directly imaged planets
        'K2-11',
        'K2-183',
        'Kepler-11',
        'Kepler-13',
        'Kepler-51',
        'LHS 1140',   # alias for GJ 3053
        'SWEEPS-11',
        'SWEEPS-4',
        'TOI 1040',
        'TOI-150.01',
        'TOI-216.01',
        'TOI-216.02',
    ]

    return targets

# --------------------------------------------------------------------

def targetlist_ariel():
    '''
    nominal Ariel target list from Edwards et al 2020
    '''

    targets = [
        '55 Cnc',
        'AU Mic',
        'CoRoT-1',
        'CoRoT-11',
        'CoRoT-19',
        'CoRoT-2',
        'CoRoT-5',
        'DS Tuc A',
        'EPIC 211945201',
        'EPIC 246851721',
        'G 9-40',
        'GJ 1132',
        'GJ 1214',
        'GJ 1252',
        'GJ 3470',
        'GJ 3473',
        'GJ 357',
        'GJ 367',
        'GJ 3929',
        'GJ 436',
        'GJ 486',
        'GJ 9827',
        'GPX-1',
        'HAT-P-1',
        'HAT-P-11',
        'HAT-P-12',
        'HAT-P-13',
        'HAT-P-14',
        'HAT-P-15',
        'HAT-P-16',
        'HAT-P-17',
        'HAT-P-18',
        'HAT-P-19',
        'HAT-P-2',
        'HAT-P-20',
        'HAT-P-21',
        'HAT-P-22',
        'HAT-P-23',
        'HAT-P-24',
        'HAT-P-25',
        'HAT-P-26',
        'HAT-P-27',
        'HAT-P-28',
        'HAT-P-29',
        'HAT-P-3',
        'HAT-P-30',
        'HAT-P-31',
        'HAT-P-32',
        'HAT-P-33',
        'HAT-P-34',
        'HAT-P-35',
        'HAT-P-36',
        'HAT-P-37',
        'HAT-P-38',
        'HAT-P-39',
        'HAT-P-4',
        'HAT-P-40',
        'HAT-P-41',
        'HAT-P-42',
        'HAT-P-43',
        'HAT-P-44',
        'HAT-P-45',
        'HAT-P-46',
        'HAT-P-49',
        'HAT-P-5',
        'HAT-P-50',
        'HAT-P-51',
        'HAT-P-52',
        'HAT-P-53',
        'HAT-P-54',
        'HAT-P-55',
        'HAT-P-56',
        'HAT-P-57',
        'HAT-P-58',
        'HAT-P-59',
        'HAT-P-6',
        'HAT-P-60',
        'HAT-P-61',
        'HAT-P-62',
        'HAT-P-64',
        'HAT-P-65',
        'HAT-P-66',
        'HAT-P-67',
        'HAT-P-68',
        'HAT-P-69',
        'HAT-P-7',
        'HAT-P-70',
        'HAT-P-8',
        'HAT-P-9',
        'HATS-1',
        'HATS-11',
        'HATS-13',
        'HATS-18',
        'HATS-2',
        'HATS-23',
        'HATS-24',
        'HATS-25',
        'HATS-26',
        'HATS-27',
        'HATS-29',
        'HATS-3',
        'HATS-30',
        'HATS-31',
        'HATS-33',
        'HATS-34',
        'HATS-35',
        'HATS-37 A',
        'HATS-38',
        'HATS-39',
        'HATS-4',
        'HATS-40',
        'HATS-41',
        'HATS-42',
        'HATS-43',
        'HATS-46',
        'HATS-47',
        'HATS-48 A',
        'HATS-5',
        'HATS-50',
        'HATS-51',
        'HATS-52',
        'HATS-53',
        'HATS-56',
        'HATS-57',
        'HATS-58 A',
        'HATS-6',
        'HATS-60',
        'HATS-62',
        'HATS-64',
        'HATS-65',
        'HATS-67',
        'HATS-68',
        'HATS-7',
        'HATS-70',
        'HATS-72',
        'HATS-9',
        'HD 106315',
        'HD 108236',
        'HD 110082',
        'HD 110113',
        'HD 136352',
        'HD 1397',
        'HD 149026',
        'HD 152843',
        'HD 15337',
        'HD 17156',
        'HD 183579',
        'HD 189733',
        'HD 191939',
        'HD 202772 A',
        'HD 207897',
        'HD 209458',
        'HD 219134',
        'HD 219666',
        'HD 221416',
        'HD 2685',
        'K2-96',    # originally was HD 3167
        'HD 332231',
        'HD 5278',
        'HD 63433',
        'HD 63935',
        'HD 73583',
        'HD 86226',
        'HD 89345',
        'HD 97658',
        'K2-93',    # originally was HIP 41378
        'HIP 65 A',
        'HIP 67522',
        'HR 858',
        'K2-107',
        'K2-116',
        'K2-121',
        'K2-129',
        'K2-132',
        'K2-136',
        'K2-138',
        'K2-139',
        'K2-140',
        'K2-141',
        'K2-155',
        'K2-18',
        'K2-19',
        'K2-198',
        'K2-222',
        'K2-232',
        'K2-237',
        'K2-238',
        'K2-239',
        'K2-24',
        'K2-25',
        'K2-260',
        'K2-261',
        'K2-266',
        'K2-28',
        'K2-280',
        'K2-284',
        'K2-287',
        'K2-289',
        'K2-29',
        'K2-295',
        'K2-3',
        'K2-31',
        'K2-32',
        'K2-329',
        'K2-333',
        'K2-334',
        'K2-34',
        'K2-353',
        'K2-36',
        'K2-39',
        'K2-403',
        'K2-405',
        'K2-406',
        'K2-52',
        'K2-55',
        'KELT-1',
        'KELT-10',
        'KELT-11',
        'KELT-12',
        'KELT-14',
        'KELT-15',
        'KELT-16',
        'KELT-17',
        'KELT-18',
        'KELT-19 A',
        'KELT-2 A',
        'KELT-20',
        'KELT-21',
        'KELT-23 A',
        'KELT-24',
        'KELT-3',
        'KELT-4 A',
        'KELT-6',
        'KELT-7',
        'KELT-8',
        'KELT-9',
        'Kepler-105',
        'Kepler-108',
        'Kepler-12',
        'Kepler-1314',
        'Kepler-1513',
        'Kepler-16',
        'Kepler-17',
        'Kepler-18',
        'Kepler-25',
        'Kepler-33',
        'Kepler-396',
        'Kepler-42',
        'Kepler-435',
        'Kepler-444',
        'Kepler-447',
        'Kepler-450',
        'Kepler-468',
        'Kepler-489',
        'Kepler-5',
        'Kepler-6',
        'Kepler-7',
        'Kepler-76',
        'Kepler-79',
        'KOI-13',
        'KOI-94',
        'KPS-1',
        'L 98-59',
        'LHS 1140',
        'LHS 1478',
        'LHS 1678',
        'LHS 3844',
        'LP 714-47',
        'LP 791-18',
        'LTT 1445 A',
        'LTT 3780',
        'LTT 9779',
        'MASCARA-1',
        'MASCARA-4',
        'NGTS-10',
        'NGTS-11',
        'NGTS-12',
        'NGTS-13',
        'NGTS-2',
        'NGTS-5',
        'NGTS-6',
        'NGTS-8',
        'pi Men',
        'Qatar-1',
        'Qatar-10',
        'Qatar-2',
        'Qatar-4',
        'Qatar-5',
        'Qatar-6',
        'Qatar-7',
        'Qatar-8',
        'Qatar-9',
        'TIC 257060897',
        'TOI-1064',
        'TOI-1075',
        'TOI-1130',
        'TOI-1201',
        'TOI-122',
        'TOI-1227',
        'TOI-1231',
        'TOI-125',
        'TOI-1259 A',
        'TOI-1260',
        'TOI-1266',
        'TOI-1268',
        'TOI-1296',
        'TOI-1298',
        'TOI-1333',
        'TOI-1411',
        'TOI-1431',
        'TOI-1442',
        'TOI-1478',
        'TOI-150.01',
        'TOI-1518',
        'TOI-157',
        'TOI-1601',
        'TOI-163',
        'TOI-1670',
        'TOI-1685',
        'TOI-169',
        'TOI-1693',
        'TOI-172',
        'TOI-1728',
        'TOI-1759',
        'TOI-178',
        'TOI-1789',
        'TOI-1807',
        'TOI-1842',
        'TOI-1860',
        'TOI-1899',
        'TOI-201',
        'TOI-2076',
        'TOI-2109',
        'TOI-216.01',
        'TOI-216.02',
        'TOI-2260',
        'TOI-2337',
        'TOI-237',
        'TOI-2411',
        'TOI-2427',
        'TOI-257',
        'TOI-2669',
        'TOI-269',
        'TOI-270',
        'TOI-3362',
        'TOI-421',
        'TOI-431',
        'TOI-4329',
        'TOI-451',
        'TOI-481',
        'TOI-500',
        'TOI-530',
        'TOI-540',
        'TOI-544',
        'TOI-559',
        'TOI-561',
        'TOI-564',
        'TOI-620',
        'TOI-628',
        'TOI-640',
        'TOI-674',
        'TOI-677',
        'TOI-700',
        'TOI-776',
        'TOI-813',
        'TOI-824',
        'TOI-833',
        'TOI-837',
        'TOI-892',
        'TOI-905',
        'TOI-954',
        'TRAPPIST-1',
        'TrES-1',
        'TrES-2',
        'TrES-3',
        'TrES-4',
        'TrES-5',
        'V1298 Tau',
        'WASP-1',
        'WASP-10',
        'WASP-100',
        'WASP-101',
        'WASP-103',
        'WASP-104',
        'WASP-105',
        'WASP-106',
        'WASP-107',
        'WASP-11',
        'WASP-110',
        'WASP-113',
        'WASP-114',
        'WASP-117',
        'WASP-118',
        'WASP-119',
        'WASP-12',
        'WASP-120',
        'WASP-121',
        'WASP-123',
        'WASP-124',
        'WASP-126',
        'WASP-127',
        'WASP-13',
        'WASP-131',
        'WASP-132',
        'WASP-133',
        'WASP-135',
        'WASP-136',
        'WASP-138',
        'WASP-139',
        'WASP-14',
        'WASP-140',
        'WASP-141',
        'WASP-142',
        'WASP-145 A',
        'WASP-147',
        'WASP-148',
        'WASP-15',
        'WASP-151',
        'WASP-153',
        'WASP-156',
        'WASP-157',
        'WASP-158',
        'WASP-159',
        'WASP-16',
        'WASP-160 B',
        'WASP-161',
        'WASP-163',
        'WASP-164',
        'WASP-165',
        'WASP-166',
        'WASP-167',
        'WASP-168',
        'WASP-169',
        'WASP-17',
        'WASP-170',
        'WASP-172',
        'WASP-173 A',
        'WASP-174',
        'WASP-175',
        'WASP-176',
        'WASP-177',
        'WASP-178',
        'WASP-18',
        'WASP-180 A',
        'WASP-181',
        'WASP-182',
        'WASP-183',
        'WASP-184',
        'WASP-185',
        'WASP-186',
        'WASP-187',
        'WASP-189',
        'WASP-19',
        'WASP-190',
        'WASP-192',
        'WASP-2',
        'WASP-20',
        'WASP-21',
        'WASP-22',
        'WASP-23',
        'WASP-24',
        'WASP-25',
        'WASP-26',
        'WASP-28',
        'WASP-29',
        'WASP-3',
        'WASP-31',
        'WASP-32',
        'WASP-33',
        'WASP-34',
        'WASP-35',
        'WASP-36',
        'WASP-37',
        'WASP-38',
        'WASP-39',
        'WASP-4',
        'WASP-41',
        'WASP-42',
        'WASP-43',
        'WASP-44',
        'WASP-45',
        'WASP-46',
        'WASP-47',
        'WASP-48',
        'WASP-49',
        'WASP-5',
        'WASP-50',
        'WASP-52',
        'WASP-53',
        'WASP-54',
        'WASP-55',
        'WASP-56',
        'WASP-57',
        'WASP-58',
        'WASP-6',
        'WASP-61',
        'WASP-62',
        'WASP-63',
        'WASP-64',
        'WASP-65',
        'WASP-66',
        'WASP-67',
        'WASP-68',
        'WASP-69',
        'WASP-7',
        'WASP-70 A',
        'WASP-71',
        'WASP-72',
        'WASP-73',
        'WASP-74',
        'WASP-75',
        'WASP-76',
        'WASP-77',  # originally WASP-77 A
        'WASP-78',
        'WASP-79',
        'WASP-8',
        'WASP-80',
        'WASP-81',
        'WASP-82',
        'WASP-83',
        'WASP-84',
        'WASP-85 A',
        'WASP-87',
        'WASP-88',
        'WASP-89',
        'WASP-90',
        'WASP-91',
        'WASP-92',
        'WASP-93',
        'WASP-94',  # originally WASP-94 A
        'WASP-95',
        'WASP-96',
        'WASP-97',
        'WASP-98',
        'WASP-99',
        'Wolf 503',
        'XO-1',
        'XO-2',    # originally XO-2 N
        'XO-3',
        'XO-4',
        'XO-5',
        'XO-6',
        'XO-7',
    ]

    return targets

# --------------------------------------------------------------------

def targetlist_G141():
    '''
    all targest with HST G141 spectra
    '''

    targets = [
    ]

    return targets

# --------------------------------------------------------------------

def targetlist_spitzer():
    '''
    all targets with Spitzer data
    '''

    targets = [
    ]

    return targets

# --------------------------------------------------------------------
def get_target_lists():
    '''
    Load in all the various target lists (a dictionary of lists)
    '''

    targetlists = {
        'active': targetlist_active(),
        'junk': targetlist_junk(),
        'roudier62': targetlist_roudier62(),
        'ariel': targetlist_ariel(),
        'G141': targetlist_G141(),
        'spitzer': targetlist_spitzer(),
        }

    return targetlists
