import time, pickle
import operator
import pandas as pd

def get_dict(train, dict_filename):
    time2 = time.time()

    row_train, col_train =  train.shape
    print 'row_train, col_train', row_train, col_train

    # before --> (word_aft1, cnt1), (word_aft2, cnt2), ..., (word_aftn, cntn)
    my_dict = {}

    # TODO: Stats session, ``training"
    for word_bef, word_aft in zip(train['before'], train['after']):
        if not (my_dict.has_key(word_bef)): # before not in
            my_dict[word_bef] = [[word_aft,1]]
        else: # before already in
            # is this word_aft already in cands?
            curr_key_pairs = my_dict[word_bef]
            FOUND_AFT = False
            for one_pair in curr_key_pairs:
                if one_pair[0] == word_aft:
                    # word_aft already in!
                    FOUND_AFT = True
                    one_pair[1] += 1
                    break
            if not FOUND_AFT:
                # word_aft not in!
                my_dict[word_bef].append([word_aft,1])

    print 'Training takes time: ', time.time() - time2

    with open(dict_filename+'.pickle','wb') as handle:
        pickle.dump(my_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_dict_fromXY(X_train, X_test, dict_filename):
    time2 = time.time()

    row_train =  len(X_train)
    print 'row_train', row_train

    # before --> (word_aft1, cnt1), (word_aft2, cnt2), ..., (word_aftn, cntn)
    my_dict = {}

    # TODO: Stats session, ``training"
    for word_bef, word_aft in zip(X_train, X_test):
        if not (my_dict.has_key(word_bef)): # before not in
            my_dict[word_bef] = [[word_aft,1]]
        else: # before already in
            # is this word_aft already in cands?
            curr_key_pairs = my_dict[word_bef]
            FOUND_AFT = False
            for one_pair in curr_key_pairs:
                if one_pair[0] == word_aft:
                    # word_aft already in!
                    FOUND_AFT = True
                    one_pair[1] += 1
                    break
            if not FOUND_AFT:
                # word_aft not in!
                my_dict[word_bef].append([word_aft,1])

    print 'Training takes time: ', time.time() - time2

    with open(dict_filename+'.pickle','wb') as handle:
        pickle.dump(my_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return my_dict

def predict_test(test, my_dict, res_filename):

    cnt = 0
    tot_num_divs = 100
    look_percents = range(1,tot_num_divs+1)
    ind_percent = 0


    row_test, col_test =  test.shape
    print 'row_test, col_test', row_test, col_test

    out = open(res_filename, "w")
    out.write('"id","after"\n')

    time3 = time.time()
    res = [] #pd.DataFrame(columns=['id','after'])
    for word_bef in test['before']:
        #sentn_id, token_id, word_bef in \
            #zip(test['sentence_id'], test['token_id'], test['before']):

        if float(cnt)/row_test > look_percents[ind_percent]/float(tot_num_divs):
            print 'Testing: %.3f has finished'%(float(cnt)/row_test)
            ind_percent += 1


        if my_dict.has_key(word_bef):
            curr_key_pairs = my_dict[word_bef]
            #max_cnt = 0
            #my_aft = None
            #for one_pair in curr_key_pairs:
            #    if one_pair[1] > max_cnt:
            #        max_cnt = one_pair[1]
            #        my_aft = one_pair[0]
            #print curr_key_pairs
            my_aft = sorted(curr_key_pairs, key=operator.itemgetter(1), reverse=True)[0][0]

        else:
            my_aft = word_bef
        res.append(my_aft)

        #res.loc[res.shape[0]] = [str(sentn_id)+'_'+str(token_id), my_aft]

        cnt += 1

    return zip(test['sentence_id'], test['token_id'], res)
    #print 'testing takes time:', time.time() - time3


def write_to_file(ress, res_filename):

    out = open(res_filename, "w")
    out.write('"id","after"\n')

    # ugly part
    for sentn_id, token_id, my_aft in ress:

        my_aft_str = str(my_aft)
        if my_aft_str.count('"')%2==1:
            #my_aft_rec = my_aft_str
            ind_tmp = my_aft_str.index('"')
            my_aft_str = my_aft_str[:ind_tmp] + '"' + my_aft_str[ind_tmp:]

            #if my_aft_rec != '"':
            #    print my_aft_rec, my_aft_str

        out.write('"'+str(sentn_id)+'_'+str(token_id) + '",')
        out.write('"'+my_aft_str+'"')
        out.write('\n')
    out.close()