# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import operator

INPUT_PATH = ''#"../input/"
SUBM_PATH = ''#"./"

import time

def solve():
    print('Train start...')
    train = open(INPUT_PATH + "en_train.csv")#, encoding='UTF8')
    line = train.readline()
    res = dict()
    total = 0
    not_same = 0

    time_start = time.time()
    while 1:
        line = train.readline().strip()

        if line == '':
            break
        total += 1
        pos = line.find('","')

        text = line[pos + 2:]

        if text[:3] == '","':
            continue
        text = text[1:-1]
        arr = text.split('","')

        if arr[0] != arr[1]:
            not_same += 1
        if arr[0] not in res:
            res[arr[0]] = dict()
            res[arr[0]][arr[1]] = 1
        else:
            if arr[1] in res[arr[0]]:
                res[arr[0]][arr[1]] += 1
            else:
                res[arr[0]][arr[1]] = 1

    train.close()

    print 'Training takes time: ', time.time() - time_start

    print('Total: {} Have diff value: {}'.format(total, not_same))

    total = 0
    changes = 0
    out = open(SUBM_PATH + 'baseline_en.csv', "w")#, encoding='UTF8')
    out.write('"id","after"\n')
    test = open(INPUT_PATH + "en_test.csv")#, encoding='UTF8')
    line = test.readline().strip()

    cnt = 0
    tot_num_divs = 100
    look_percents = range(1,tot_num_divs+1)
    ind_percent = 0
    row_test, col_test = 1088564, 3
    while 1:
        if float(cnt)/row_test > look_percents[ind_percent]/float(tot_num_divs):
            print 'Testing: %.3f has finished'%(float(cnt)/row_test)
            ind_percent += 1

        line = test.readline().strip()
        if line == '':
            break

        pos = line.find(',')
        i1 = line[:pos]
        line = line[pos + 1:]

        pos = line.find(',')
        i2 = line[:pos]
        line = line[pos + 1:]

        line = line[1:-1]
        out.write('"' + i1 + '_' + i2 + '",')
        if line in res:
            #print res[line].items()
            srtd = sorted(res[line].items(), key=operator.itemgetter(1), reverse=True)
            #print line
            #print srtd
            #raw_input()

            out.write('"' + srtd[0][0] + '"')
            changes += 1
        else:
            out.write('"' + line + '"')

        out.write('\n')
        total += 1

        cnt += 1

    print('Total: {} Changed: {}'.format(total, changes))
    test.close()
    out.close()


solve()