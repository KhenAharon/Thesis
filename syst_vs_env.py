import random
import dynet as dy
import numpy as np

# to use a particular example, uncomment its line and comment all the others

#from cycle_abc import *
# from cycle_random_abc import *
# from cycle_random_abc_anchor1 import *
# from cycle_random_abc_anchor2 import *
# from cycle_random_abc_anchor3 import *
# from cycle_random_abc_memory import *
# from detector import *
#from cycle_abb import *  # "permitted"
#from schedule import *
#from cases import *
#from unlucky import *
#from strategy import *
# from strategy2 import *
# from cycle_abcbc import *
# from cycle_abcbca import *
#from combination_lock import *
# from combination_lock2 import *
# from choice import *
# from choice2 import *
# from choice3 import *
# from choice4 import *
# from choice5 import *
# from choice6 import *
# from choice_tournament import *
#from choice_scc import *
#from least_failures import *
#from good_failure import *
# from choice_and_good_failure import *
# from match_coins import *
#from matching_simplified import *
# from combine_scc_cycle import *
# from combine_scc_cycle_simplified import *
# from combine_scc_cycle_simplified_2 import *
#from combine_schedule_cycle import *

# number of controlled tests after training, and length of control executions
C = 100
L_C = 200
model = "any_available"


def count_failures(execution):
    counter = 0
    for step in execution:
        if step[0][:4] == "fail":
            counter += 1
    return counter


def default_lookahead(history):
    return 0


def random_lookahead(history, average=3, variation=2):
    return random.randint(average - 2, average + 2)


def run(pve, print_first=False, print_probs=False):
    pve.reinitialize()
    if print_first:
        print(pve.generate_controlled_execution(50, print_probs=print_probs)[0][0])
    else:
        print(pve.generate_controlled_execution(50, print_probs=print_probs))


def count_results_within_bound(results, bound):
    return len([r for r in results if r < bound])


# same to test on a large amount of trainings
def test(number_of_tests, number_of_runs, size, print_probs=False, random_exploration=False, new_loss=False,
         lookahead=1, epsilon=0, compare_loss=False):
    results = []
    T = number_of_runs
    L = size

    for test in range(number_of_tests):
        pve = create(model)
        for training in range(T):
            for length in range(1, L):
                pve.reinitialize()
                pve.generate_training_execution(length, print_probs=False, random_exploration=random_exploration,
                                                new_loss=new_loss, lookahead=lookahead, epsilon=epsilon,
                                                compare_loss=compare_loss, discount_loss = False,discount_factor = 1)

        failures = []
        for control in range(C):
            pve.reinitialize()
            execution = pve.generate_controlled_execution(L_C, print_probs=print_probs)
            failures.append(count_failures(execution))
        percentage = 0
        for i in range(C):
            percentage += (failures[i]/L_C) * 100
        percentage /= C

        results.append(percentage)

    average = 0
    for r in results:
        average += r
    return average/len(results)


for l in [0, 3, 20]:
    for e in [0, 0.2, 0.5]:
        print("lookahead = ", l, "epsilon = ", e)
        print("new_loss", test(50, 50, 50, new_loss=True, lookahead=l, epsilon=e, compare_loss=False))
        print("old_loss", test(50, 50, 50, new_loss=True, lookahead=l, epsilon=e, compare_loss=True))

# random
pve_rand = create(model)
failures = []
for control in range(C):
    pve_rand.reinitialize()
    execution = pve_rand.generate_random_execution(L_C)
    failures.append(count_failures(execution))
percentage = 0
for i in range(C):
    percentage += failures[i]/L_C
percentage /= C
print("(random)", percentage * 100, "%")
