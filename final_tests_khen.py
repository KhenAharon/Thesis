import random
import dynet as dy
import numpy as np
from all_examples import *


# If using a non-combined example, uncomment the next three lines
number_of_epochs = 50 #this is what we called T so far
training_execution_length = 50 #this is what we called L so far
runs = 1 # will be >1 only for combined examples

# If using a combined example, comment the three lines number_of_epochs and training_execution_length and runs above, and uncomment the ones below that correspond to the example:
#number_of_epochs = 1000 #this is what we called T so far
#training_execution_length = 8 #this is what we called L so far
#runs = 2

#from combine_scc_cycle_simplified import *
#number_of_epochs = 100 #this is what we called T so far
#training_execution_length = 8 #this is what we called L so far
#runs = 5

# number of controlled tests after training, and length of control executions
C = 100
L_C = 200
L_C_short = 50
model = "any_available"


def count_failures(execution):
    counter = 0
    for step in execution:
        if step[0][:4] == "fail":
            counter += 1
    return counter


for figure_number in range(2, 3):
    line = ""
    results = []
    results_short = []

    best = L_C
    best_short = L_C_short

    worst = 0
    worst_short = 0

    print("\nFigure", figure_number)
    if figure_number == 1:
        pve = create_fig1_abc(model)  # this generates the system and environment
        pve_rand = create_fig1_abc(model)
    elif figure_number == 2:
        pve = create_fig2_schedule(model)
        pve_rand = create_fig2_schedule(model)
    elif figure_number == 3:
        pve = create_fig3_cases(model)
        pve_rand = create_fig3_cases(model)
    elif figure_number == 4:
        pve = create_fig4_unlucky(model)
        pve_rand = create_fig4_unlucky(model)
    elif figure_number == 5:
        pve = create_fig5_least_failures(model)
        pve_rand = create_fig5_least_failures(model)
    elif figure_number == 6:
        pve = create_fig6_good_failure(model)
        pve_rand = create_fig6_good_failure(model)
    elif figure_number == 7:
        pve = create_fig7_combination_lock(model)
        pve_rand = create_fig7_combination_lock(model)
    elif figure_number == 8:
        pve = create_fig8_choice_scc(model)
        pve_rand = create_fig8_choice_scc(model)
    elif figure_number == 9:
        pve = create_fig9_strategy(model)
        pve_rand = create_fig9_strategy(model)
    elif figure_number == 10:
        pve = create_fig10_schedule_cycle(model)
        pve_rand = create_fig10_schedule_cycle(model)
    elif figure_number == 11:
        pve = create_fig11_cycle_scc(model)
        pve_rand = create_fig11_cycle_scc(model)

    for (lookahead,epsilon) in [(0, 0), (0, 0.2), (0, 0.5), (3, 0), (3, 0.2), (3, 0.5), (20, 0), (20, 0.2), (20, 0.5)]:
        for tests in range(10):
            pve.reinitialize()
            pve.create_RNN()

            for _ in range(runs):
                    # iterating over the epochs
                    for epoch in range(number_of_epochs):
                        d_epsilon = epsilon * (1 - epoch / (number_of_epochs - 1))
                        # iterating over the training sequences from length 1 to length L
                        for length in range(1,training_execution_length):

                            pve.reinitialize()  # return system and environment to initial states

                            # Now we generate a training sequence.
                            pve.generate_training_execution(length,lookahead = lookahead,epsilon = epsilon,compare_loss = False)

            failures = []
            failures_short = []

            for control in range(C):
                pve.reinitialize()
                execution = pve.generate_controlled_execution(L_C_short,print_probs = False)
                failures_short.append(count_failures(execution))

            for control in range(C):
                pve.reinitialize()
                execution = pve.generate_controlled_execution(L_C,print_probs = False)
                failures.append(count_failures(execution))

            percentage = 0
            percentage_short = 0
            for i in range(C):
                percentage += failures[i]/L_C
                percentage_short += failures_short[i]/L_C_short
            percentage /= C
            percentage_short /= C

            if percentage > worst:
                worst = percentage
            if percentage < best:
                best = percentage
            if percentage_short > worst_short:
                worst_short = percentage_short
            if percentage_short < best_short:
                best_short = percentage_short

            results.append(percentage)
            results_short.append(percentage_short)
            # test number..

        # global results for each set of parameters
        add = ""
        add += str(best_short*100) + " " + str(worst_short*100) + " " + str(sum(results_short)/len(results_short)*100) + " "
        add += str(best*100) + " " + str(worst*100) + " " + str(sum(results)/len(results)*100) + " "
        print(add)
        line += add
    # all global results for each figure

    # random
    failures = []
    for control in range(C):
        pve_rand.reinitialize()
        execution = pve_rand.generate_random_execution(L_C)
        failures.append(count_failures(execution))
    percentage = 0
    for i in range(C):
        percentage += failures[i]/L_C
    percentage /= C
    line += str(percentage * 100) + " "

    print("***********final***********:\n")
    print(line)


def run(pve, steps = 50, print_first=False, print_probs=False):
    pve.reinitialize()
    if print_first:
        print(pve.generate_controlled_execution(steps, print_probs=print_probs)[0][0])
    else:
        print(pve.generate_controlled_execution(steps, print_probs=print_probs))



# same to test on a large amount of trainings
def test(number_of_tests, number_of_runs, size, print_probs=False, random_exploration=False, new_loss=False,
         lookahead=1, epsilon=0, compare_loss=False):
    results = []
    T = number_of_runs
    L = size

    for test in range(number_of_tests):
        for T in [number_of_runs]:
            for L in [size]:
                pve = create(model)
                for training in range(T):
                    for length in range(1,L):
                        pve.reinitialize()
                        pve.generate_training_execution(length,print_probs = False,random_exploration = random_exploration,new_loss = new_loss,lookahead = T,epsilon = epsilon,compare_loss = compare_loss)
                        
                        #print("____________")
                    for length in range(1,L):
                        pve.reinitialize()
                        pve.generate_training_execution(length,print_probs = False,random_exploration = random_exploration,new_loss = new_loss,lookahead = 0,epsilon = epsilon,compare_loss = compare_loss)
                        
                failures = []
                for control in range(C):
                    pve.reinitialize()
                    execution = pve.generate_controlled_execution(L_C,print_probs = print_probs)
                    #print("____________")
                    failures.append(count_failures(execution))
                percentage = 0
                for i in range(C):
                    percentage += (failures[i]/L_C)*100
                percentage /= C
                #print("test number",test+1,"(",T,",",L,")",percentage*100,"%")
                #run(pve,True)
                run(pve,False,print_probs = True)

        
        results.append(percentage)

    average = 0
    for r in results:
        average += r
    return average/len(results)



# =============================================================================
# for l in [0, 3, 20]:
#     for e in [0, 0.2, 0.5]:
#         print("lookahead = ", l, "epsilon = ", e)
#         print("new_loss", test(50, 50, 50, new_loss=True, lookahead=l, epsilon=e, compare_loss=False))
#         print("old_loss", test(50, 50, 50, new_loss=True, lookahead=l, epsilon=e, compare_loss=True))
# =============================================================================
