
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

def simulate_rolling_dice(p=0.5, num_trials=50, random_state=0):
    simul_res = []
    np.random.seed(random_state)
    for i in range(num_trials):
        cur_rand = np.random.random()
        if cur_rand >= p:
            simul_res.append(1)
        else:
            simul_res.append(0)

    return simul_res

def t_test(binomial_array, theo_prob=0.5, alpha=0.05):
    '''
    Steps:
    1.Define Null and Alter Hypothesis:
        - See if the probability of 1 in binomial_array (prob_data=np.mean(binomial_array))
            is the same as theo_prob.
        - Null: yes (prob_data = theo_prob). Alter: no (prob_data != theo_prob)
    2.Alpha:
        - alpha (two side)
    3.df:
        - len(binomial_array)-1
    4.Decision rule:
        - get t score boundaries from t_bound = stats.ppf(1-alpha/2., df)
        - rule is t_stat \in (-t_bound, t_bound)
    5.Calc t_stat
        - t = (prob_data-theo_prob)/[(sample standard error)/sqrt(n)]
        where sample standard error =

    :param binomial_array:
    :param theo_prob:
    :param alpha:
    :return:
    '''

    from scipy.stats import t

    n = len(binomial_array)
    df = n-1
    '''
    This is wrong:
    mean = np.mean(binomial_array)
    std = np.std(binomial_array)
    
    Essentially categorical!
    '''
    data_prob = np.mean(binomial_array)
    std = np.sqrt(n*theo_prob*(1-theo_prob)) # np.sqrt(n*data_prob*(1-data_prob))

    print 'n:', n
    print 'data_prob and theo_prob:', data_prob, theo_prob

    '''
    Should I use t or z test here?! I suppose t should always work!
    '''
    t_stat = (data_prob-theo_prob)/(std/np.sqrt(n))
    print 'actual t score:', t_stat

    t_stat_bound = t.ppf(1-alpha/2., df=df) # Percent point function, quantile function
    print 't score range: (%.2f, %.2f)' % (-t_stat_bound, t_stat_bound)

    print

    #TODO: finish later...


if __name__ == '__main__':
    for num_trials in [5,50,500,5000, 50000, 500000]:
        simul_res = simulate_rolling_dice(p=0.55, num_trials=num_trials, random_state=23)
        t_test(simul_res, theo_prob=0.5)
