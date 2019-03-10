
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

def main_ttest():
    for num_trials in [5,50,500,5000, 50000, 500000]:
        simul_res = simulate_rolling_dice(p=0.55, num_trials=num_trials, random_state=23)
        t_test(simul_res, theo_prob=0.5)

def simulate_clone_competitions(alp = 0.1,
                                mu=10.,
                                r=9.,
                                H=200,
                                tot_time=100,
                                random_state=0):
    '''
    alp * H = (mu-r) * N

    1. Set the time {\displaystyle t=0} t=0.
    2. Choose an initial state k.
    3. Get the number {\displaystyle N_{k}} N_{k} of all possible transition rates, from state k into a generic state i.
    4. Find the candidate event to carry out i by uniformly sampling from the {\displaystyle N_{k}} N_{k} transitions above.
    5. Accept the event with probability {\displaystyle f_{ki}=r_{ki}/r_{0}} f_{ki}=r_{ki}/r_{0}, where {\displaystyle r_{0}} r_{0} is a suitable upper bound for {\displaystyle r_{ki}} r_{ki}. It is often easy to find {\displaystyle r_{0}} r_{0} without having to compute all {\displaystyle r_{ki}} r_{ki} (e.g., for Metropolis transition rate probabilities).
    6. If accepted, carry out event i (update the current state {\displaystyle k\rightarrow i} k\rightarrow i).
    7. Get a new uniform random number {\displaystyle u^{\prime }\in (0,1]} u^{\prime }\in (0,1].
    8. Update the time with {\displaystyle t=t+\Delta t} t=t+\Delta t, where {\displaystyle \Delta t=(N_{k}r_{0})^{-1}\ln(1/u^{\prime })} \Delta t=(N_{k}r_{0})^{-1}\ln(1/u^{\prime }).
    9. Return to step 3.

    :return:
    '''
    print 'alp, H, r, mu:', alp, H, r, mu

    t = 0
    times_clones = [(t, [0]*H)] # clones[i][j]: i-th time pt, j-th clone
    N = sum(times_clones[-1][1])

    np.random.seed(random_state)

    while t < tot_time:
        clones = times_clones[-1][1][:] # avoiding changing them

        tot_rate = (r+mu)*N + alp*H
        rand1 = np.random.random() # [0.0, 1.0)
        t += np.log(1./(1-rand1))/tot_rate

        rand2 = np.random.random()
        tester = rand2 * tot_rate

        happened = False
        for j in range(H):
            # which clone's turn?
            if happened:
                break
            for rate_effect in [(alp+clones[j]*r, 1),
                                (clones[j]*mu, -1)]:
                # which event?
                rate, effect = rate_effect
                if tester <= rate:
                    clones[j] += effect
                    N += effect
                    happened = True
                    break
                else:
                    tester -= rate
        times_clones.append((t, clones))
    return times_clones

def main_Gillespie():
    alp = 0.1
    mu = 10.
    r = 9.
    H = 100
    print 'N^*=', alp*H/(mu-r)

    tot_time = 100

    times_clones = simulate_clone_competitions(alp=alp,
                                mu=mu,
                                r=r,
                                H=H,
                               tot_time=tot_time,
                               random_state=50)
    t_s = []
    N_s = []

    '''
    Mean is not necessarily equal to N*! 
    Get more samples when N is large!!!
    '''
    t_selected = np.linspace(0, tot_time, num=100)
    t_selected_ind = 0
    N_selected = []
    for time_clones in times_clones:
        t = time_clones[0]
        N = sum(time_clones[1])

        t_s.append(t)
        N_s.append(N)

        if t_selected_ind < len(t_selected) and t > t_selected[t_selected_ind]:
            N_selected.append(N)
            t_selected_ind += 1

    print 'np.mean(N_s):', np.mean(N_s)
    print 'Adjusted N mean:', np.mean(N_selected)

    plt.plot(t_s, N_s)
    plt.show()

if __name__ == '__main__':
    main_Gillespie()

