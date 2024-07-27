'''target targetlists ds'''
import excalibur
import os
import csv
import logging; log = logging.getLogger(__name__)

# --------------------------------------------------------------------
def get_target_lists():
    '''
    Load in all the various target lists that are compiled here
    '''

    # target list sizes as of March 6, 2024:
    #  active: 818
    #  roudier62: 60
    #  G141:      86
    #  arielMCS_Nov2023_transit:     300
    #  arielMCS_Nov2023_maxVisits25: 330
    #  arielMCS_Feb2024_transit:     226
    #  arielMCS_Feb2024_maxVisits25: 270

    targetlists = {
        'active': targetlist_active(),
        'roudier62': targetlist_roudier62(),
        'G141': targetlist_G141(),
        'JWST': targetlist_JWST(),
        'Spitzer': targetlist_Spitzer(),
        'arielMCS_Nov2023_transit': targetlist_ArielMCSknown('Nov2023',transitCategoryOnly=True),
        'arielMCS_Nov2023_maxVisits25': targetlist_ArielMCSknown('Nov2023',maxVisits=25),
        'arielMCS_Feb2024_transit': targetlist_ArielMCSknown('Feb2024',transitCategoryOnly=True),
        'arielMCS_Feb2024_maxVisits25': targetlist_ArielMCSknown('Feb2024',maxVisits=25),
    }

    # for targetlist in targetlists:
    #     print('# of targets:',targetlist,len(targetlists[targetlist]))

    return targetlists
# --------------------------------------------------------------------
def targetlist_active():
    '''
    all good targets (everything except target/edit.py/dropouts)
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
        'Kepler-11',
        'Kepler-12',
        'Kepler-125',
        'Kepler-126',
        'Kepler-127',
        'Kepler-13',
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
        'HATS-22',
        'K2-30',
        'Kepler-1308',
        'Qatar-3',
        'WASP-129',
        'WASP-144',
        'Kepler-51',
        'WD 1856',
        'GJ 4102',
        'HD 80606',
        'GJ 4332',
        'LTT 5972',
        'GJ 3090',
        'GJ 806',
        'HD 109833',
        'HD 207496',
        'HD 260655',
        'HD 93963 A',
        'HD 95338',
        'HIP 29442',
        'HIP 94235',
        'HIP 97166',
        'K2-105',
        'K2-370',
        'K2-415',
        'K2-60',
        'Kepler-1656',
        'Kepler-470',
        'Kepler-63',
        'Kepler-96',
        'NGTS-14 A',
        'TOI-1136',
        'TOI-1272',
        'TOI-1278',
        'TOI-1288',
        'TOI-132',
        'TOI-139',
        'TOI-1422',
        'TOI-1468',
        'TOI-1470',
        'TOI-1634',
        'TOI-1694',
        'TOI-1695',
        'TOI-1710',
        'TOI-1801',
        'TOI-181',
        'TOI-1853',
        'TOI-1859',
        'TOI-199',
        'TOI-2000',
        'TOI-2010',
        'TOI-2018',
        'TOI-2134',
        'TOI-2136',
        'TOI-220',
        'TOI-2364',
        'TOI-2443',
        'TOI-2445',
        'TOI-2459',
        'TOI-2498',
        'TOI-251',
        'TOI-277',
        'TOI-3082',
        'TOI-332',
        'TOI-3629',
        'TOI-3785',
        'TOI-4010',
        'TOI-444',
        'TOI-4479',
        'TOI-4600',
        'TOI-4641',
        'TOI-470',
        'TOI-5126',
        'TOI-532',
        'TOI-5344',
        'TOI-5398',
        'TOI-5678',
        'TOI-5704',
        'TOI-5803',
        'TOI-672',
        'TOI-712',
        'TOI-908',
        'TOI-913',
        'TOI-942',
        'TOI-969',
        'Wolf 327',
        'CoRoT-3',
        'CoRoT-36',
        'Gaia-1',
        'Gaia-2',
        'HATS-10',
        'HATS-12',
        'HATS-45',
        'HATS-55',
        'HATS-61',
        'HD 118203',
        'HD 15906',
        'HD 21749',
        'HD 235088',
        'HD 28109',
        'HIP 113103',
        'HIP 116454',
        'HIP 9618',
        'K2-233',
        'K2-240',
        'K2-285',
        'K2-321',
        'K2-344',
        'K2-348',
        'K2-399',
        'K2-417',
        'K2-99',
        'Kepler-1515',
        'Kepler-1517',
        'Kepler-1658',
        'Kepler-411',
        'Kepler-91',
        'KOI-12',
        'NGTS-24',
        'NGTS-9',
        'PH2',
        'TIC 237913194',
        'TOI-1107',
        'TOI-1181',
        'TOI-1194',
        'TOI-1246',
        'TOI-1338',
        'TOI-1408',
        'TOI-1416',
        'TOI-1420',
        'TOI-1452',
        'TOI-1516',
        'TOI-1680',
        'TOI-1811',
        'TOI-1820',
        'TOI-1937 A',
        'TOI-198',
        'TOI-2025',
        'TOI-2046',
        'TOI-2048',
        'TOI-206',
        'TOI-2081',
        'TOI-2145',
        'TOI-2152 A',
        'TOI-2154',
        'TOI-2158',
        'TOI-2193 A',
        'TOI-2194',
        'TOI-2202',
        'TOI-2207',
        'TOI-2236',
        'TOI-2338',
        'TOI-2421',
        'TOI-2497',
        'TOI-2524',
        'TOI-2567',
        'TOI-2570',
        'TOI-2583 A',
        'TOI-2587 A',
        'TOI-262',
        'TOI-2641',
        'TOI-2796',
        'TOI-2803 A',
        'TOI-2818',
        'TOI-2842',
        'TOI-2977',
        'TOI-3023',
        'TOI-3235',
        'TOI-3331 A',
        'TOI-3364',
        'TOI-3540 A',
        'TOI-3688 A',
        'TOI-3693',
        'TOI-3714',
        'TOI-3807',
        'TOI-3819',
        'TOI-3884',
        'TOI-3912',
        'TOI-3976 A',
        'TOI-4087',
        'TOI-411',
        'TOI-4137',
        'TOI-4145 A',
        'TOI-4308',
        'TOI-4342',
        'TOI-4377',
        'TOI-4406',
        'TOI-4463 A',
        'TOI-4551',
        'TOI-4603',
        'TOI-4791',
        'TOI-615',
        'TOI-622',
        'TOI-778',
        'TOI-858',
        'WASP-130',
        'WASP-150',
        'WASP-162',
        'WASP-171',
        'WASP-193',
        'WASP-60',
        'HD 110067',
        'TOI-3261',
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

def targetlist_G141():
    '''
    all targets with HST G141 spectra.  currently 89 stars
    '''

    targets = [
        '55 Cnc',
        'AU Mic',
        'CoRoT-1',  # STARE, not SCAN
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
        'HAT-P-2',
        'HAT-P-24',
        'HAT-P-26',
        'HAT-P-3',
        'HAT-P-32',
        'HAT-P-38',
        'HAT-P-41',
        'HAT-P-7',
        'HAT-P-70',
        'HATS-7',
        'HD 106315',
        'HD 149026',
        'HD 189733',
        'HD 191939',
        'HD 209458',
        'HD 219666',
        'HD 97658',
        'K2-18',
        'K2-24',
        'K2-3',
        'K2-33',
        'K2-93',
        'K2-96',
        'KELT-1',
        'KELT-11',
        'KELT-20',
        'KELT-7',
        'KELT-9',
        'Kepler-11',  # STARE, not SCAN
        'Kepler-13',  # STARE, not SCAN
        'Kepler-138',
        'Kepler-16',
        # 'Kepler-1625',  # G141 is STARE only, not SCAN
        # 'Kepler-51',  # G141 is STARE only, not SCAN
        'Kepler-79',
        'L 98-59',
        # 'LHS 1140',  # alias for GJ 3470
        # 'LHS 6343',  # has G141, but is a false-positive candidate planet
        'LTT 1445 A',
        'LTT 9779',
        # 'TIC 184892124',  # G141 is STARE only, not SCAN
        'TOI-1201',
        'TOI-1231',
        'TOI-1759',
        'TOI-270',
        'TOI-431',
        'TOI-561',
        'TOI-674',
        'TRAPPIST-1',
        'TrES-4',  # STARE, not SCAN
        'V1298 Tau',
        'WASP-101',
        'WASP-103',
        'WASP-107',
        'WASP-117',
        'WASP-12',
        'WASP-121',
        'WASP-127',
        'WASP-17',
        'WASP-178',
        'WASP-18',
        'WASP-19',
        'WASP-29',
        'WASP-31',
        'WASP-33',
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
        'WASP-77',
        'WASP-79',
        'WASP-80',
        'WASP-96',
        'WASP-98',
        'XO-1',
        'XO-2',
        'HAT-P-14',
        'HD 86226']

    return targets

# --------------------------------------------------------------------
def targetlist_JWST():
    '''
    all targets with JWST spectra.  currently XX stars
    '''

    targets = [
        # '55 Cnc',
        # 'GJ 1214',
        'CoRoT-5',
        'GJ 357',
        'GJ 367',
        'GJ 4102',
        'GJ 4332',
        'GJ 486',
        'HAT-P-14',
        'HAT-P-24',
        'HAT-P-70',
        'HATS-72',
        'HD 15337',
        'HD 191939',
        'HD 260655',
        'HD 80606',
        'HD 86226',
        'HIP 67522',
        'K2-33',
        'KELT-9',
        'Kepler-138',
        'Kepler-293',
        'Kepler-444',
        'Kepler-51',
        'L 98-59',
        'LP 791-18',
        'LTT 1445 A',
        'LTT 5972',
        'LTT 9779',
        'NGTS-10',
        'TOI-270',
        'TOI-776',
        'TOI-3714',
        'WASP-107',
        'WASP-117',
        'WASP-127',
        'WASP-163',
        'WASP-166',
        'WASP-178',
        'WASP-25',
        'WASP-54',
        'WASP-96',
        'WASP-98',
        'GJ 486',   # NIRSPEC data.timing
        # 'GJ 1132',  # NIRSPEC data.timing
        'WASP-19',  # NIRISS data.timing
        'WASP-39',  # NIRISS data.timing
    ]

    return targets

# --------------------------------------------------------------------

def targetlist_Spitzer():
    '''
    all targets with Spitzer lightcurves.  currently 187 stars
    '''

    targets = [
        '55 Cnc',
        'AU Mic',
        'CoRoT-2',
        'CoRoT-5',  # new
        'DS Tuc A',
        'GJ 1132',
        'GJ 1214',
        'GJ 1252',
        'GJ 3053',
        'GJ 3470',
        'GJ 357',  # new
        'GJ 367',  # new
        'GJ 4332',  # new
        'GJ 436',
        'GJ 486',  # new
        'GJ 4102',  # new
        'GJ 9827',
        'HAT-P-1',
        'HAT-P-11',
        'HAT-P-12',
        'HAT-P-13',
        'HAT-P-14',  # new
        'HAT-P-15',
        'HAT-P-17',
        'HAT-P-18',
        'HAT-P-19',
        'HAT-P-2',
        'HAT-P-20',
        'HAT-P-22',
        'HAT-P-23',
        'HAT-P-24',  # new
        'HAT-P-26',
        'HAT-P-3',
        'HAT-P-30',
        'HAT-P-32',
        'HAT-P-33',
        'HAT-P-34',
        'HAT-P-38',
        'HAT-P-40',
        'HAT-P-41',
        'HAT-P-6',
        'HAT-P-7',
        'HAT-P-70',  # new
        'HAT-P-8',
        'HATS-3',
        'HATS-7',
        'HATS-72',  # new
        'HD 15337',  # new
        'HD 106315',
        'HD 149026',
        'HD 189733',
        'HD 191939',  # new
        'HD 202772 A',
        'HD 260655',  # new
        'HD 209458',
        'HD 213885',
        'HD 219134',
        'HD 219666',
        'HD 23472',
        'HD 80606',  # new
        'HD 86226',  # new
        'HD 97658',
        'HIP 67522',  # new
        'HIP 116454',  # new
        'HR 858',
        'K2-132',
        'K2-136',
        'K2-138',
        'K2-141',
        'K2-167',
        'K2-174',
        'K2-18',
        'K2-19',
        'K2-21',
        'K2-212',
        'K2-24',
        'K2-25',
        'K2-28',
        'K2-3',
        'K2-32',
        'K2-33',
        'K2-79',
        'K2-96',
        'K2-97',
        'KELT-1',
        'KELT-11',
        'KELT-14',
        'KELT-16',
        'KELT-2 A',
        'KELT-20',
        'KELT-3',
        'KELT-7',
        'KELT-9',
        'Kepler-138',
        'Kepler-1485',
        'Kepler-18',
        'Kepler-293',  # new
        'Kepler-32',
        'Kepler-37',
        'Kepler-444',  # new
        'Kepler-45',
        'Kepler-68',
        'Kepler-9',
        'Kepler-93',
        'LHS 1140',
        'LHS 3844',
        'L 98-59',  # new
        'LP 791-18',
        'LTT 1445 A',  # new
        'MASCARA-1',
        'Qatar-1',
        'Qatar-2',
        'TOI-270',
        'TOI-700',
        'TOI-776',  # new
        'TRAPPIST-1',
        'TrES-2',
        'TrES-3',
        'V1298 Tau',
        'WASP-1',
        'WASP-10',
        'WASP-100',
        'WASP-101',
        'WASP-104',
        'WASP-107',
        'WASP-11',
        'WASP-117',  # new
        'WASP-12',
        'WASP-121',
        'WASP-127',
        'WASP-13',
        'WASP-131',
        'WASP-14',
        'WASP-140',
        'WASP-15',
        'WASP-16',
        'WASP-163',  # new
        'WASP-166',  # new
        'WASP-178',  # new
        'WASP-17',
        'WASP-18',
        'WASP-19',
        'WASP-21',
        'WASP-25',  # new
        'WASP-29',
        'WASP-3',
        'WASP-31',
        'WASP-33',
        'WASP-34',
        'WASP-36',
        'WASP-38',
        'WASP-39',
        'WASP-4',
        'WASP-43',
        'WASP-46',
        'WASP-49',
        'WASP-50',
        'WASP-52',
        'WASP-54',  # new
        'WASP-6',
        'WASP-62',
        'WASP-63',
        'WASP-64',
        'WASP-65',
        'WASP-67',
        'WASP-69',
        'WASP-7',
        'WASP-72',
        'WASP-74',
        'WASP-75',  # new
        'WASP-76',
        'WASP-77',
        'WASP-78',
        'WASP-79',
        'WASP-8',
        'WASP-80',
        'WASP-87',
        'WASP-94',
        'WASP-95',
        'WASP-96',
        'WASP-97',
        'WASP-98',  # new
        'Wolf 503',
        'XO-1',
        'XO-2',
        'XO-3',
        'XO-4',
    ]

    return targets

# --------------------------------------------------------------------

def read_ArielMCS_info(filename='Ariel_MCS_Known_2024-02-14.csv'):
    '''
    Load in the MCS table with all the Ariel target info
    The most recent file is 2/14/24, but also consider the 11/20/23 file
    '''

    arielDir = excalibur.context['data_dir']+'/ariel/'

    listofDictionaries = []

    if not os.path.isfile(arielDir + filename):
        log.warning('--< PROBLEM: Ariel MCS table not found >--')
    else:
        with open(arielDir + filename, 'r', encoding='ascii') as file:
            csvFile = csv.DictReader(file)

            # print('starting to read',filename)
            for line in csvFile:
                listofDictionaries.append(line)

    return listofDictionaries
# --------------------------------------------------------------------

def targetlist_ArielMCSknown(filedate='Nov2023',
                             maxVisits=666,
                             transitCategoryOnly=False):
    '''
    Select a batch of targets from the Ariel MCS list of known planets

    If transitCategoryOnly, include all planets in the 'transit' or 'either' category
    Otherwise make the selection based on the number of transit visits (<= maxVisits)
    '''

    if filedate=='Nov2023':
        filename = 'Ariel_MCS_Known_2023-11-20.csv'
    elif filedate=='Feb2024':
        filename = 'Ariel_MCS_Known_2024-02-14.csv'
    else:
        log.warning('--< PROBLEM: Unknown date for the Ariel MCS table >--')
        filename = 'Ariel_MCS_Known_2023-11-20.csv'

    targetinfo = read_ArielMCS_info(filename=filename)

    aliases = arielAliases()

    targetList = []
    for target in targetinfo:
        selected = False
        if transitCategoryOnly:
            if target['Preferred Method']=='Transit' or target['Preferred Method']=='Either':
                selected = True
        else:
            if float(target['Tier 2 Transits']) <= maxVisits:
                selected = True

        if selected:
            # special case for TOI-216.01 and TOI-216.02
            if target['Planet Name'].endswith('.01'):
                target['Planet Name'] = target['Planet Name'][:-3] + ' b'
                # print('fixing planet name from .01 to',target['Planet Name'])
            elif target['Planet Name'].endswith('.02'):
                target['Planet Name'] = target['Planet Name'][:-3] + ' c'
                # print('fixing planet name from .02 to',target['Planet Name'])

            # translate a handful of MCS target names to Excalibur aliases
            if target['Star Name'] in aliases:
                alias = aliases[target['Star Name']]
                target['Star Name'] = alias
                target['Planet Name'] = alias + target['Planet Name'][-2:]

            # save the star names (not the planet names).  that's what's needed for pipeline call
            # targetList.append(target['Planet Name'])
            if target['Star Name'] in targetList:
                # print('skipping a multi-planet entry for',target['Star Name'])
                pass
            else:
                targetList.append(target['Star Name'])

    return targetList
# --------------------------------------------------------------------
def arielAliases():
    '''
    Some of the names in Ariel MCS are different from in Excalibur.
    Translate them using these aliases.
    (first column is MCS name, second column is Excalibur name)
    '''

    aliases = {
        'HD 3167':'K2-96',
        'L 168-9':'GJ 4332',
        'LHS 1140':'GJ 3053',
        'LHS 475':'GJ 4102',
        'TOI-836':'LTT 5972',
        'WASP-94 A':'WASP-94',
        'GJ 143':'HD 21749',
        'HIP 41378':'K2-93',
    }

    return aliases
# --------------------------------------------------------------------
