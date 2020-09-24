import random
import dynet as dy
import numpy as np


def softmax(l):
    soft = l.copy()
    expl = np.exp(l)
    for i in range(len(l)):
        soft[i] = expl[i] / np.sum(expl)
    return soft


class Transition:
    def __init__(self, name, start, end):
        self.name = name
        self.start = start
        self.end = end


class process:
    # finite automaton where transitions can be shared with other processes.
    def __init__(self, name, states=[], internal=[], shared=[], initial_state=None, update_states=[]):
        self.name = name
        self.states = states  # list of names of the states.
        self.initial_state = initial_state
        self.current_state = initial_state
        self.internal = internal  # transitions that are specific to the process.
        self.shared = shared  # transitions of both processes, initialized empty and filled by add_transition.
        self.internal_transitions = []  # same as above, but the names of the transitions.
        self.shared_transitions = []  # initialized as empty lists that will be filled using list_transitions.
        self.all_transitions = []  # all_transitions will be updated as the concatenation of the two previous lists.
        self.update_states = update_states  # only states on which a learning pass can be done. empty = all states.

    def add_state(self, name):
        self.states.append(name)
        if self.current_state is None:
            self.initial_state = name
            self.current_state = name
            
    def define_update_state(self, name):
        if name not in self.states:
            raise ValueError("no state named", name)
        else:
            self.update_states.append(name)  # new update possible state, if it is a valid state of the process.
            
    def list_update_states(self, name_list):
        for name in name_list:
            self.define_update_state(name)  # define a list of update possible states.
            
    def reinitialize(self):
        self.current_state = self.initial_state  # return to initial state.
        if self.update_states is []:  # if list of update states is undefined, all states will be updated.
            self.list_update_states(self.states)
            
    def add_transition(self, name, start, end, internal=True):
        if internal:
            self.internal.append(Transition(name, start, end))  # add a new transition from state start to state end.
        else:
            self.shared.append(Transition(name, start, end))
        
    def trigger_transition(self, tr_name):
        try:  # move the process according to the transition tr_name, printing an error if not possible.
            self.current_state = next(tr.end for tr in self.internal if tr_name == tr.name and tr.start == self.current_state)
        except StopIteration:
            try:  # we trigger it in internal, but if not exist, in shared.
                self.current_state = next(tr.end for tr in self.shared if tr_name == tr.name and tr.start == self.current_state)
            except StopIteration:
                print("No transition named", tr_name, "from state", self.current_state)
      
    def list_transitions(self):
        for tr in self.internal:
            if tr.name not in self.internal_transitions:  # update the names of currently defined transitions.
                self.internal_transitions.append(tr.name)
        for tr in self.shared:
            if tr.name not in self.shared_transitions:
                self.shared_transitions.append(tr.name)
        self.all_transitions = self.internal_transitions + self.shared_transitions
        
    def available_transitions(self):
        available = []
        for tr in self.internal + self.shared:
            if tr.name not in available and tr.start == self.current_state:
                available.append(tr.name)
        return available  # returns a list of names of transitions that can be triggered in the current state.


class System:
    def __init__(self, name, processes):  # a system is a compound of processes.
        self.name = name
        self.processes = processes  # a list of the processes in the compound.
        self.shared_transitions = []  # list of the shared transitions of the different processes
        self.networks = None  # Neural networks associated to the processes (listed in the same order as the processes)
        self.R = None  # R, bias, parameters and trainer are respective lists of the parameters for the NNs
        self.bias = None
        self.parameters = None
        self.trainer = None
        
    def reinitialize(self):
        for pr in self.processes:  # reinitialize all processes in the system
            pr.reinitialize()
    
    def get_process(self, name):  # returns the process with that name
        return next(proc for proc in self.processes if name == proc.name)
    
    def add_process(self, process):
        self.processes.append(process)  # add a new process to the system
        
    def add_transition(self, name, pr_list, start_list, end_list):
        # add a new transition shared between processes in pr_list
        # for process pr_list[i], the transition goes from state start_list[i] to state end_list[i]
        if len(pr_list) == 1:
            start = start_list[0]
            end = end_list[0]
            for j in range(len(start)):
                self.get_process(pr_list[0]).add_transition(name, start[j], end[j])
        else:
            for i in range(len(pr_list)):
                start = start_list[i]
                end = end_list[i]
                for j in range(len(start)):
                    self.get_process(pr_list[i]).add_transition(name, start[j], end[j], False)
            self.shared_transitions.append((name, pr_list, start_list, end_list))


class plant_environment(System):
    def __init__(self, name, plant: process, environment: process, model="correctly_guess", layers=1, hidden_dim=5):
        self.plant = plant  # define the system with processes plant and environment
        self.environment = environment
        System.__init__(self, name, [self.plant, self.environment])
        self.plant.list_transitions()
        self.environment.list_transitions()  # @added

        self.layers = layers  # parameters of the neural network that will be used by the plant
        self.hidden_dim = hidden_dim
        self.model = model

    # Below are two functions that define the RNN, that the plant will use
    # Input structure: n*k dimensions,
    # with dimension j+i for 0 <= i <= k-1 and 0 <= j <= n-1 representing success/failure (+1/-1) of action j in state i
    def create_RNN(self):
        # create an RNN using the parameters of the system, with the VanillaLSTMBuilder
        # from DyNet, and a SimpleSGDTrainer, with input structure defined above
        self.plant.list_transitions()
        self.environment.list_transitions()  # @added

        self.parameters = dy.ParameterCollection()
        self.parameters2 = dy.ParameterCollection()  # @added

        input_dim = (len(self.plant.internal_transitions)+len(self.plant.shared_transitions))*len(self.plant.states)
        output_dim = len(self.plant.all_transitions)

        input_dim2 = (len(self.environment.internal_transitions)+len(self.environment.shared_transitions))*len(self.environment.states)
        output_dim2 = len(self.environment.all_transitions)

        self.R = self.parameters.add_parameters((output_dim,self.hidden_dim))
        self.bias = self.parameters.add_parameters((output_dim))
        self.network = dy.VanillaLSTMBuilder(self.layers,input_dim,self.hidden_dim,self.parameters,forget_bias = 1.0)
        self.trainer = dy.SimpleSGDTrainer(self.parameters)

        self.R2 = self.parameters2.add_parameters((output_dim2, self.hidden_dim))
        self.bias2 = self.parameters2.add_parameters((output_dim2))
        self.network2 = dy.VanillaLSTMBuilder(self.layers, input_dim2, self.hidden_dim, self.parameters2, forget_bias=1.0)
        self.trainer2 = dy.SimpleSGDTrainer(self.parameters2)
    
    def RNN_input(self, last_transition, is_plant=True):
        # converts the last transition that was triggered into a valid input for the LSTM
        # if there was no previous transition (first step), everything is put to 0
        # else the input is modified to put +1 or -1 in the entry corresponding
        # to the last transition and the current state, depending on the success/failure
        # of the last transition
        if is_plant:
            v = [0]*((len(self.plant.internal_transitions)+len(self.plant.shared_transitions))*len(self.plant.states))
            i = next(i for i in range(len(self.plant.states)) if self.plant.states[i] == self.plant.current_state)
        else:
            v = [0]*((len(self.environment.internal_transitions)+len(self.environment.shared_transitions))*len(self.environment.states))
            i = next(i for i in range(len(self.environment.states)) if self.environment.states[i] == self.environment.current_state)

        if last_transition == None:
            pass
        else:
            if last_transition[:4] == "fail":
                failed_action = ""
                current_char_index = 5
                while last_transition[current_char_index] != ")":
                        failed_action += last_transition[current_char_index]
                        current_char_index += 1
                if is_plant:
                    j = next(j for j in range(len(self.plant.shared_transitions)) if failed_action == self.plant.shared_transitions[j])
                    v[((len(self.plant.internal_transitions) + j))*len(self.plant.states)+i] = -1
                else:
                    j = next(j for j in range(len(self.environment.shared_transitions)) if failed_action == self.environment.shared_transitions[j])
                    v[((len(self.environment.internal_transitions) + j))*len(self.environment.states)+i] = -1
            else:
                if is_plant:
                    j = next(j for j in range(len(self.plant.all_transitions)) if last_transition == self.plant.all_transitions[j])
                    v[(len(self.plant.internal_transitions) + j)*len(self.plant.states)+i] = 1
                else:
                    j = next(j for j in range(len(self.environment.all_transitions)) if last_transition == self.environment.all_transitions[j])
                    v[(len(self.environment.internal_transitions) + j)*len(self.environment.states)+i] = 1
        return v       
        

        
# Three different ways to interpret the output : softmax passes the output of the network through a softmax to interpret it as a probability distribution over the transitions, argmax chooses the transition with highest value of the output, random interprets the output directly as a distribution (usually using random).       
        
    def RNN_output(self,output,method = "random", print_probs = False, is_plant=True):
        #this takes as input the output from the RNN, and returns the next transition
        #to trigger according to the chosen method for interpreting the output
        #the transition is guaranteed to be available to the plant in the current state
        
        next_transition = ""
        if is_plant:
            available = self.plant.available_transitions()
        else:
            available = self.environment.available_transitions()

        if method == "softmax":
            if is_plant:
                output = softmax([output[i] for i,tr in enumerate(self.plant.all_transitions) if tr in available])
            else:
                output = softmax([output[i] for i,tr in enumerate(self.environment.all_transitions) if tr in available])

            next_transition = random.choices(available,output)[0]
            if print_probs and len(output) >1:
                print(([(output[i]) for i in range(len(output))]))
        if method == "argmax":
            max_value = -float("inf")
            max_index = -1
            if is_plant:
                for i,tr in enumerate(self.plant.all_transitions):
                    if tr in available:
                        if output[i] > max_value:
                            max_value = output[i]
                            max_index = i
                next_transition = self.plant.all_transitions[max_index]
            else:
                for i, tr in enumerate(self.environment.all_transitions):
                    if tr in available:
                        if output[i] > max_value:
                            max_value = output[i]
                            max_index = i
                next_transition = self.environment.all_transitions[max_index]

        elif method == "random":
            if is_plant:
                next_transition = random.choices(available,([output[i] for i,tr in enumerate(self.plant.all_transitions) if tr in available]))[0]
            else:
                next_transition = random.choices(available,([output[i] for i,tr in enumerate(self.environment.all_transitions) if tr in available]))[0]
            
            if print_probs and len(available) >1:
                if is_plant:
                    print(([(tr,output[i]) for i,tr in enumerate(self.plant.all_transitions) if tr in available]))
                else:
                    print(([(tr,output[i]) for i,tr in enumerate(self.environment.all_transitions) if tr in available]))

        if print_probs and len(available) >1:
            pass
            #print("choice:",next_transition)
        return next_transition
    
    
    

 
    def check_transition(self,some_transition,environment_transition = None,some_strategy = None, is_plant_transition=True): # strategy was env_strateg, some_transition was plant_transition
        
        #given a transition proposed by the plant, check if the environment can comply
        #return a pair [p,e] where:
        #if environment complies, p = e = plant_transition
        #else, p = fail(plant_transition) and e is a random transition available to the environment
        #possibility to use a specific strategy for the environment instead of a random choice

        if is_plant_transition:
            available = self.environment.available_transitions()
        else:
            available = self.plant.available_transitions()

        if some_transition in available:
            return [some_transition,some_transition]
        else:
            if some_strategy == None:
                return ["fail("+some_transition+")",random.choice(available)]
            else:
                if is_plant_transition:
                    return["fail("+some_transition+")",some_strategy(self.environment)]
                return["fail("+some_transition+")",some_strategy(self.plant)]





    def trigger_transition(self,transition, is_plant=True):
        #takes as input the pair [p,e] from check_transition, triggers p if 
        #it is not failed, and triggers e.
        if transition[0][:4] == "fail":
            pass
        else:
            if is_plant:
                self.plant.trigger_transition(transition[0])
            else:
                self.environment.trigger_transition(transition[0])
        if is_plant:
            self.environment.trigger_transition(transition[1])
        else:
            self.plant.trigger_transition(transition[1])

    
    
    
    def random_transition(self):
        #choose a random transition for the plant among the available ones
        
        plant_action = random.choice([tr for tr in self.plant.internal + self.plant.shared if tr.start == self.plant.current_state]).name
        environment_action = random.choice([tr for tr in self.environment.internal + self.environment.shared if tr.start == self.environment.current_state]).name
        return [plant_action,environment_action]          
    
    
    def generate_random_execution(self,steps,some_strategy = None, is_plant=True):
        #generate an execution of size steps of the compound plant/environment
        #where the plant always chooses randomly its actions among the available ones
        #returns the execution under the form of a list of transitions or failed transitions
        
        execution = []
        for s in range(steps):
            tr = self.random_transition()
            if is_plant:
                tr = self.check_transition(tr[0],environment_strategy = some_strategy, is_plant_transition=True)
            else:
                tr = self.check_transition(tr[0], environment_transition=some_strategy, is_plant_transition=False)
            self.trigger_transition(tr, is_plant=is_plant)
            execution.append(tr)
        return execution
    



    def generate_controlled_execution(self,steps,environment_strategy = None,print_probs = False, is_plant=True):
        
        #generates an execution where the plant choose its actions according to
        #its associated RNN, without any learning taking place.
        
        #print_probs is here for debugging purposes : prints the probability to
        #choose each action at each step
        
        #initialization
        execution = []
        dy.renew_cg()
        if is_plant:
            state = self.network.initial_state()
        else:
            state = self.network2.initial_state()
        last_transition = None
        output = []
        
        for step in range(steps):
            
            #giving new input to the LSTM
            network_input = self.RNN_input(last_transition, is_plant=is_plant)
            input_vector = dy.inputVector(network_input)
            state = state.add_input(input_vector)
            
            #computation of the output: the LSTM unit is followed by a linear layer with
            #bias, then the result is passed through a softmax
            if is_plant:
                output = dy.softmax(self.R*state.output() + self.bias).value()
            else:
                output = dy.softmax(self.R2*state.output() + self.bias2).value()
            
            #computation of the next action proposed by the plant
            next_action = self.RNN_output(output,print_probs = print_probs, is_plant=is_plant)
            
            
# =============================================================================
#             if environment_strategy == None:
#                 environment_action = random.choice([tr for tr in self.environment.internal + self.environment.shared if tr.start == self.environment.current_state]).name
#             else:
#                 environment_action = environment_strategy(self.environment)
# =============================================================================
            
            #check transition and compute environment action
            tr = self.check_transition(next_action,some_strategy=environment_strategy, is_plant_transition=is_plant)
            if print_probs:
                print("choice:",tr)
            
            #trigger transitions and complete the execution
            self.trigger_transition(tr, is_plant=is_plant)
            execution.append(tr)
            last_transition = tr[0]
            
        return execution


    
    def generate_training_execution(self,steps,lookahead = 1,lookahead_function = None,environment_strategy = None,print_probs = False,random_exploration = False,random_step = 4,new_loss = True,epsilon = 0,compare_loss = False,discount_loss = False,discount_factor = 1, is_plant=True):
        
        #generates a training execution: an execution is generated, but with a training
        #pass being done at every step
        
        #print_probs is again here for debugging purposes
        
        #when lookahead == 0, there is no lookahead and the learning depends only on
        #the success/failure of the last transition
        #when lookahead > 0, the information of successes/failures is stored in
        #rollout and rollout_error lists, and used after lookahead steps to do a training pass
        
        #the value of epsilon is the probability that the plant will choose an action
        #randomly at every step instead of following its RNN's advice
        
        #random_exploration and random_step allow to start the execution randomly for
        #random_steps steps before using the RNN for advice.
        
        rollout = [None]*(lookahead+1)
        rollout_error = [None]*(lookahead+1)
        
        
        def rollout_update(rollout,new_state):
            return rollout[1:]+[new_state]
            
        def rollout_error_update(rollout_errors,error):
            return rollout_errors[1:]+[error]


        #initialization
        execution = []
        if is_plant:
            state = self.network.initial_state()
        else:
            state = self.network2.initial_state()
        last_transition = None
        output = None
        output_value = None
        
        past_loss = 0
        nb_fail = 0
        
        dy.renew_cg()
        if is_plant:
            state = self.network.initial_state()
        else:
            state = self.network2.initial_state()

        
        
        loss = [dy.scalarInput(0)]
        
        
        for step in range(steps):
            
            
            #new input for the RNN
            network_input = self.RNN_input(last_transition, is_plant=is_plant)

            input_vector = dy.inputVector(network_input)
            state = state.add_input(input_vector)
            
            #computation of the output
            if is_plant:
                output = dy.softmax(self.R*state.output() + self.bias)
            else:
                output = dy.softmax(self.R2*state.output() + self.bias2)

            output_value = output.value()
            
            
            #compute the next action for the plant depending on the chosen strategy
            if not random_exploration or step > random_step:
                if random.random() < epsilon:
                    next_some_action = self.random_transition()[0]
                else:
                    next_some_action = self.RNN_output(output_value,print_probs = print_probs, is_plant=is_plant)
            else:
                if is_plant:
                    next_some_action = random.choice(self.plant.available_transitions())
                else:
                    next_some_action = random.choice(self.environment.available_transitions())

            
# =============================================================================
#                 if environment_strategy == None:
#                     environment_action = random.choice([tr for tr in self.environment.internal + self.environment.shared if tr.start == self.environment.current_state]).name
#                 else:
#                     environment_action = environment_strategy(self.environment)
# =============================================================================
                
            #check if environment can comply and if not, choose an action for it    
            tr = self.check_transition(next_some_action,some_strategy=environment_strategy, is_plant_transition=is_plant)
            
            #update the information for the loss with lookahead: remember the successes and failures
            plant_tr = tr[0]
            if plant_tr[:4] == "fail":
                plant_tr = plant_tr[5]
                if not random_exploration or step > random_step:
                    rollout_error = rollout_error_update(rollout_error,True)
            else:
                if not random_exploration or step > random_step:
                    rollout_error = rollout_error_update(rollout_error,False)
            
            
            #store information about the output to compute the loss
            if not random_exploration or step > random_step:
                if is_plant:
                    rollout = rollout_update(rollout,(output,self.plant.available_transitions(),plant_tr))
                else:
                    rollout = rollout_update(rollout,(output,self.environment.available_transitions(),plant_tr))

            
            
            
            
            
            #now for the training. default is learning with our usual loss; if compare_loss is True, we use the reinforce-loss instead; if discount_loss is true, we use the usual loss with given discount factor. Prints an error message if both compare_loss and discount_loss are True.
            
            if not random_exploration or step > random_step:
                
                
                if not compare_loss and not discount_loss: #using our usual loss. Using Yoav's modification is
                                        # triggered by putting new_loss = True
                    i_train = next(i for i in range(len(rollout)) if rollout[i] != None)
                    if rollout[i_train] != None:
                        
                        #count the amount of successes and failures in the lookahead window
                        nb_failures = rollout_error.count(True)
                        nb_successes = 1+lookahead-nb_failures
                        
                        #print(nb_failures,nb_successes)
                        
                        #compute the loss (usually using new_loss, which is the loss described in the paper)
                        if is_plant:
                            my_len = len(self.plant.all_transitions)
                        else:
                            my_len = len(self.environment.all_transitions)

                        for i in range(my_len):
                            if is_plant:
                                cur_transition=self.plant.all_transitions[i]
                            else:
                                cur_transition=self.environment.all_transitions[i]

                            if cur_transition in rollout[i_train][1]:
                                if cur_transition == rollout[i_train][2]:#chosen action
                                    loss.append((nb_successes/(lookahead+1))*dy.pickneglogsoftmax(rollout[i_train][0],i))
                                    #print(loss[-1].value())
                                    
                                    
                                # original loss
                                if not new_loss:
                                    loss.append((nb_failures/(lookahead+1))*dy.pickneglogsoftmax(rollout[i_train][0],i))
                                ###############
                                
                                # Yoav's correction of the loss
                                else:
                                    if cur_transition != rollout[i_train][2]:#not chosen action
                                        loss.append((nb_failures/(lookahead+1))*dy.pickneglogsoftmax(rollout[i_train][0],i))
                                        #print(loss[-1].value())
                                ###############
    
                        
                    loss_compute = dy.esum(loss) #* (1+nb_fail)
    
                    #Do the training pass
                    loss_compute.value()
                    loss_compute.backward()
                    if is_plant:
                        self.trainer.update()
                    else:
                        self.trainer2.update()
                    
                    loss = [dy.scalarInput(0)]    
            
            
            
            
            
            
                elif compare_loss and not discount_loss: #this is for using a REINFORCE-like loss. Instead of
                                # reinforcing every non-chosen action when there is a failure,
                                # we use a negative reinforcement on the chosen action
                                # for successes, reinforcement works the same as in the usual loss
                    i_train = next(i for i in range(len(rollout)) if rollout[i] != None)
                    if rollout[i_train] != None:
                        
                        #count the amount of successes and failures in the lookahead window
                        nb_failures = rollout_error.count(True)
                        nb_successes = lookahead-nb_failures
                        
                        #compute the loss (usually using new_loss, which is the loss described in the paper)
                        if is_plant:
                            my_len = len(self.plant.all_transitions)
                        else:
                            my_len = len(self.environment.all_transitions)

                        for i in range(my_len):
                            if is_plant:
                                cur_transition = self.plant.all_transitions[i]
                            else:
                                cur_transition = self.environment.all_transitions[i]

                            if cur_transition in rollout[i_train][1]:
                                if cur_transition == rollout[i_train][2]:#chosen action
                                    loss.append((nb_successes/(lookahead+1))*dy.pickneglogsoftmax(rollout[i_train][0],i))
                                    loss.append(-(nb_failures/(lookahead+1))*dy.pickneglogsoftmax(rollout[i_train][0],i))
                                    
                                    
    
                        
                    loss_compute = dy.esum(loss) #* (1+nb_fail)
    
                    #Do the training pass
                    loss_compute.value()
                    loss_compute.backward()
                    if is_plant:
                        self.trainer.update()
                    else:
                        self.trainer2.update()
                    
                    loss = [dy.scalarInput(0)]    
            
            
                
                
                
                elif not compare_loss and discount_loss: #using discounted loss. Using Yoav's modification is
                                        # triggered by putting new_loss = True
                    i_train = next(i for i in range(len(rollout)) if rollout[i] != None)
                    if rollout[i_train] != None:
                        
                        #count the amount of successes and failures with discount factor in the lookahead window
                        discounted_failures = 0
                        discounted_successes = 0
                        current_discount = 1
                        for i in range(len(rollout_error)):
                            discounted_failures += current_discount if not rollout_error[i] else 0
                            discounted_successes += current_discount if rollout_error[i] else 0
                            current_discount *= discount_factor
                            
                            #print(discounted_failures,discounted_successes)
                        
                        #compute the loss (usually using new_loss, which is the loss described in the paper)
                        if is_plant:
                            my_len = len(self.plant.all_transitions)
                        else:
                            my_len = len(self.environment.all_transitions)

                        for i in range(my_len):
                            if is_plant:
                                cur_transition = self.plant.all_transitions[i]
                            else:
                                cur_transition = self.environment.all_transitions[i]

                            if cur_transition in rollout[i_train][1]:
                                if cur_transition == rollout[i_train][2]:#chosen action
                                    loss.append((discounted_successes/(lookahead+1))*dy.pickneglogsoftmax(rollout[i_train][0],i))
                                    #print(loss[-1].value())
                                    
                                    
                                # original loss
                                if not new_loss:
                                    loss.append((discounted_failures)*dy.pickneglogsoftmax(rollout[i_train][0],i))
                                ###############
                                
                                # Yoav's correction of the loss
                                else:
                                    if cur_transition != rollout[i_train][2]:#not chosen action
                                        loss.append((discounted_failures/(lookahead+1))*dy.pickneglogsoftmax(rollout[i_train][0],i))
                                        print(loss[-1].value())
                                ###############
    
                        
                    loss_compute = dy.esum(loss) #* (1+nb_fail)
    
                    #Do the training pass
                    loss_compute.value()
                    loss_compute.backward()

                    if is_plant:
                        self.trainer.update()
                    else:
                        self.trainer2.update()
                    
                    loss = [dy.scalarInput(0)]  
                    
                    
                    
                else:
                    print("compare_loss and discount_loss are incompatible")
            
            
            
            #to be used when using update_states, not used for now
            if is_plant:
                cur_state = self.plant.current_state
                update_states= self.plant.update_states
            else:
                cur_state = self.environment.current_state
                update_states= self.environment.update_states


            if cur_state in update_states:
                
                pass
                nb_fail = 0
                
                #loss = [dy.scalarInput(0)]
                
            #trigger transitions and count the total amount of failures
            #(total amount not used anymore, was used to weight the loss)
            self.trigger_transition(tr)
            #print(tr)
            if tr[0][:4] == "fail":
                nb_fail += 1
                past_loss = past_loss +1
# =============================================================================
#                 else:
#                     nb_fail = 0
# =============================================================================

            execution.append(tr)
            last_transition = tr[0]
        return(execution)
            
            
            
# =============================================================================
#     def default_lookahead(self,history):
#         return 0
# =============================================================================
        
        
        
        
        
    def generate_training_execution_target_network(self,steps,lookahead = 1,lookahead_function = None,environment_strategy = None,print_probs = False,random_exploration = False,random_step = 4,new_loss = False,epsilon = 0,compare_loss = False,discount_loss = False,discount_factor = 1):
        
        #generates a training execution: an execution is generated, but with a training
        #pass being done at every step
        
        #print_probs is again here for debugging purposes
        
        #when lookahead == 0, there is no lookahead and the learning depends only on
        #the success/failure of the last transition
        #when lookahead > 0, the information of successes/failures is stored in
        #rollout and rollout_error lists, and used after lookahead steps to do a training pass
        
        #the value of epsilon is the probability that the plant will choose an action
        #randomly at every step instead of following its RNN's advice
        
        #random_exploration and random_step allow to start the execution randomly for
        #random_steps steps before using the RNN for advice.
        
        rollout = [None]*(lookahead+1)
        rollout_error = [None]*(lookahead+1)
        
        
        def rollout_update(rollout,new_state):
            return rollout[1:]+[new_state]
            
        def rollout_error_update(rollout_errors,error):
            return rollout_errors[1:]+[error]


        #initialization
        execution = []
        state = self.network.initial_state()
        state_target = self.network.initial_state()
        last_transition = None
        output = None
        output_value = None
        
        past_loss = 0
        nb_fail = 0
        
        dy.renew_cg()
        state = self.network.initial_state()
        state_target = self.network.initial_state()
        
        
        loss = [dy.scalarInput(0)]
        
        
        for step in range(steps):
            
            
            #new input for the RNN
            network_input = self.RNN_input(last_transition)

            input_vector = dy.inputVector(network_input)
            input_vector_target = dy.inputVector(network_input)
            
            state = state.add_input(input_vector)
            state_target = state_target.add_input(input_vector)
            
            #computation of the output
            output = dy.softmax(self.R*state.output() + self.bias)
            output_target = dy.softmax(self.R*state_target.output() + self.bias)
            
            output_value = output.value()
            output_target_value = output_target.value()
            
            
            #compute the next action for the plant depending on the chosen strategy
            if not random_exploration or step > random_step:
                if random.random() < epsilon:
                    next_plant_action = self.random_transition()[0]
                else:
                    next_plant_action = self.RNN_output(output_target_value,print_probs = print_probs)
            else:
                next_plant_action = random.choice(self.plant.available_transitions())
            
# =============================================================================
#                 if environment_strategy == None:
#                     environment_action = random.choice([tr for tr in self.environment.internal + self.environment.shared if tr.start == self.environment.current_state]).name
#                 else:
#                     environment_action = environment_strategy(self.environment)
# =============================================================================
                
            #check if environment can comply and if not, choose an action for it    
            tr = self.check_transition(next_plant_action,environment_strategy=environment_strategy)
            
            #update the information for the loss with lookahead: remember the successes and failures
            plant_tr = tr[0]
            if plant_tr[:4] == "fail":
                plant_tr = plant_tr[5]
                if not random_exploration or step > random_step:
                    rollout_error = rollout_error_update(rollout_error,True)
            else:
                if not random_exploration or step > random_step:
                    rollout_error = rollout_error_update(rollout_error,False)
            
            
            #store information about the output to compute the loss
            if not random_exploration or step > random_step:
                rollout = rollout_update(rollout,(output_target,self.plant.available_transitions(),plant_tr))
            
            
            
            
            
            #now for the training. default is learning with our usual loss; if compare_loss is True, we use the reinforce-loss instead; if discount_loss is true, we use the usual loss with given discount factor. Prints an error message if both compare_loss and discount_loss are True.
            
            if not random_exploration or step > random_step:
                
                
                if not compare_loss and not discount_loss: #using our usual loss. Using Yoav's modification is
                                        # triggered by putting new_loss = True
                    i_train = next(i for i in range(len(rollout)) if rollout[i] != None)
                    if rollout[i_train] != None:
                        
                        #count the amount of successes and failures in the lookahead window
                        nb_failures = rollout_error.count(True)
                        nb_successes = 1+lookahead-nb_failures
                        
                        #print(nb_failures,nb_successes)
                        
                        #compute the loss (usually using new_loss, which is the loss described in the paper)
                        for i in range(len(self.plant.all_transitions)):
                            if self.plant.all_transitions[i] in rollout[i_train][1]:
                                if self.plant.all_transitions[i] == rollout[i_train][2]:#chosen action
                                    loss.append((nb_successes/(lookahead+1))*dy.pickneglogsoftmax(rollout[i_train][0],i))
                                    #print(loss[-1].value())
                                    
                                    
                                # original loss
                                if not new_loss:
                                    loss.append((nb_failures/(lookahead+1))*dy.pickneglogsoftmax(rollout[i_train][0],i))
                                ###############
                                
                                # Yoav's correction of the loss
                                else:
                                    if self.plant.all_transitions[i] != rollout[i_train][2]:#not chosen action
                                        loss.append((nb_failures/(lookahead+1))*dy.pickneglogsoftmax(rollout[i_train][0],i))
                                        #print(loss[-1].value())
                                ###############
    
                        
                    loss_compute = dy.esum(loss) #* (1+nb_fail)
    
                    #Do the training pass
                    loss_compute.value()
                    loss_compute.backward()
                    
                    self.trainer.update()
                    
                    loss = [dy.scalarInput(0)]    
            
            
            
            
            
            
                elif compare_loss and not discount_loss: #this is for using a REINFORCE-like loss. Instead of
                                # reinforcing every non-chosen action when there is a failure,
                                # we use a negative reinforcement on the chosen action
                                # for successes, reinforcement works the same as in the usual loss
                    i_train = next(i for i in range(len(rollout)) if rollout[i] != None)
                    if rollout[i_train] != None:
                        
                        #count the amount of successes and failures in the lookahead window
                        nb_failures = rollout_error.count(True)
                        nb_successes = lookahead-nb_failures
                        
                        #compute the loss (usually using new_loss, which is the loss described in the paper)
                        for i in range(len(self.plant.all_transitions)):
                            if self.plant.all_transitions[i] in rollout[i_train][1]:
                                if self.plant.all_transitions[i] == rollout[i_train][2]:#chosen action
                                    loss.append((nb_successes/(lookahead+1))*dy.pickneglogsoftmax(rollout[i_train][0],i))
                                    loss.append(-(nb_failures/(lookahead+1))*dy.pickneglogsoftmax(rollout[i_train][0],i))
                                    
                                    
    
                        
                    loss_compute = dy.esum(loss) #* (1+nb_fail)
    
                    #Do the training pass
                    loss_compute.value()
                    loss_compute.backward()
                    
                    self.trainer.update()
                    
                    loss = [dy.scalarInput(0)]    
            
            
                
                
                
                elif not compare_loss and discount_loss: #using discounted loss. Using Yoav's modification is
                                        # triggered by putting new_loss = True
                    i_train = next(i for i in range(len(rollout)) if rollout[i] != None)
                    if rollout[i_train] != None:
                        
                        #count the amount of successes and failures with discount factor in the lookahead window
                        discounted_failures = 0
                        discounted_successes = 0
                        current_discount = 1
                        for i in range(len(rollout_error)):
                            discounted_failures += current_discount if not rollout_error[i] else 0
                            discounted_successes += current_discount if rollout_error[i] else 0
                            current_discount *= discount_factor
                            
                            #print(discounted_failures,discounted_successes)
                        
                        #compute the loss (usually using new_loss, which is the loss described in the paper)
                        for i in range(len(self.plant.all_transitions)):
                            if self.plant.all_transitions[i] in rollout[i_train][1]:
                                if self.plant.all_transitions[i] == rollout[i_train][2]:#chosen action
                                    loss.append((discounted_successes/(lookahead+1))*dy.pickneglogsoftmax(rollout[i_train][0],i))
                                    #print(loss[-1].value())
                                    
                                    
                                # original loss
                                if not new_loss:
                                    loss.append((discounted_failures)*dy.pickneglogsoftmax(rollout[i_train][0],i))
                                ###############
                                
                                # Yoav's correction of the loss
                                else:
                                    if self.plant.all_transitions[i] != rollout[i_train][2]:#not chosen action
                                        loss.append((discounted_failures/(lookahead+1))*dy.pickneglogsoftmax(rollout[i_train][0],i))
                                        print(loss[-1].value())
                                ###############
    
                        
                    loss_compute = dy.esum(loss) #* (1+nb_fail)
    
                    #Do the training pass
                    loss_compute.value()
                    loss_compute.backward()
                    
                    self.trainer.update()
                    
                    loss = [dy.scalarInput(0)]  
                    
                    
                    
                else:
                    print("compare_loss and discount_loss are incompatible")
            
            
            
            #to be used when using update_states, not used for now
            if self.plant.current_state in self.plant.update_states:
                
                pass
                nb_fail = 0
                
                #loss = [dy.scalarInput(0)]
                
            #trigger transitions and count the total amount of failures
            #(total amount not used anymore, was used to weight the loss)
            self.trigger_transition(tr)
            #print(tr)
            if tr[0][:4] == "fail":
                nb_fail += 1
                past_loss = past_loss +1
# =============================================================================
#                 else:
#                     nb_fail = 0
# =============================================================================

            execution.append(tr)
            last_transition = tr[0]
        return(execution)
            
            
      

        
            
            
            
    def generate_training_execution_varying_lookahead(self,steps,lookahead_function,environment_strategy = None,print_probs = False,random_exploration = False,random_step = 4,new_loss = False,epsilon = 0):
        
        
        #same as before, but instead of a fixed lookahead value, we have a lookahead
        #function that takes as input the history of the system and returns the
        #value of the lookahead at this point
        
        #for now, only used with a random lookahead between fixed bounds
        
        
        rollout = []
        rollout_error = []
        history = [(None,self.plant.current_state)]
        remaining_lookahead = []
        lookahead = []
        
        
        
        def rollout_update(rollout,new_state):
            return rollout+[new_state]
            
        def rollout_error_update(rollout_errors,error):
            return rollout_errors+[error]

        if self.model == "any_available":
            execution = []
            state = self.network.initial_state()
            last_transition = None
            output = None
            output_value = None
            
            past_loss = 0
            nb_fail = 0
            
            dy.renew_cg()
            state = self.network.initial_state()
            
            
            loss = [dy.scalarInput(0)]
            
            
            for step in range(steps):
                
                network_input = self.RNN_input(last_transition)

                input_vector = dy.inputVector(network_input)
                state = state.add_input(input_vector)
                output = dy.softmax(self.R*state.output() + self.bias)
                
                
                
                output_value = output.value()
                
                if not random_exploration or step > random_step:
                    if random.random() < epsilon:
                        next_plant_action = self.random_transition()[0]
                    else:
                        next_plant_action = self.RNN_output(output_value,print_probs = print_probs)
                else:
                    next_plant_action = random.choice(self.plant.available_transitions())
                
                if environment_strategy == None:
                    environment_action = random.choice([tr for tr in self.environment.internal + self.environment.shared if tr.start == self.environment.current_state]).name
                else:
                    environment_action = environment_strategy(self.environment)
                    
                    
                tr = self.check_transition(next_plant_action,environment_action,environment_strategy)
                plant_tr = tr[0]
                if plant_tr[:4] == "fail":
                    plant_tr = plant_tr[5]
                    if not random_exploration or step > random_step:
                        rollout_error = rollout_error_update(rollout_error,True)
                else:
                    if not random_exploration or step > random_step:
                        rollout_error = rollout_error_update(rollout_error,False)
                    
                if not random_exploration or step > random_step:
                    rollout = rollout_update(rollout,(output,self.plant.available_transitions(),plant_tr))
                
                
                
                available = self.plant.available_transitions()
                
                remaining_lookahead.append(lookahead_function(history))
                lookahead.append(lookahead_function(history))
                
                if not random_exploration or step > random_step:
                
                    for (i_train,remaining) in enumerate(remaining_lookahead):
                        if remaining == 0:
                            
                    
                            if rollout[i_train] != None:
                                
                                
                                nb_failures = rollout_error[i_train:].count(True)
                                nb_successes = lookahead[i_train]-nb_failures
                                
                                for i in range(len(self.plant.all_transitions)):
                                    if self.plant.all_transitions[i] in rollout[i_train][1]:
                                        if self.plant.all_transitions[i] == rollout[i_train][2]:#chosen action
                                            loss.append((nb_successes/(lookahead[i_train]+1))*dy.pickneglogsoftmax(rollout[i_train][0],i))
                                            
                                            
                                        # original loss
                                        if not new_loss:
                                            loss.append((nb_failures/(lookahead[i_train]+1))*dy.pickneglogsoftmax(rollout[i_train][0],i))
                                        ###############
                                        
                                        # Yoav's correction of the loss
                                        else:
                                            if self.plant.all_transitions[i] != rollout[i_train][2]:#not chosen action
                                                loss.append((nb_failures/(lookahead[i_train]+1))*dy.pickneglogsoftmax(rollout[i_train][0],i))
                                        ###############
        
                                
                            #print("training over",execution,"with",nb_failures,"failures")
                            loss_compute = dy.esum(loss) #* (1+nb_fail)
            
                
                            loss_compute.value()
                            #print(loss_compute.value())
                            loss_compute.backward()
                            self.trainer.update()
                            
                            loss = [dy.scalarInput(0)] 
                
                for i in range(len(remaining_lookahead)):
                    remaining_lookahead[i] -= 1
                    
                if self.plant.current_state in self.plant.update_states:
                    
                    pass
                    nb_fail = 0
                    
                    #loss = [dy.scalarInput(0)]
                    
                    
                self.trigger_transition(tr)
                #print(tr)
                if tr[0][:4] == "fail":
                    nb_fail += 1
                    past_loss = past_loss +1
# =============================================================================
#                 else:
#                     nb_fail = 0
# =============================================================================

                execution.append(tr)
                last_transition = tr[0]
                history.append((last_transition,self.plant.current_state))
            return(execution)
            
            
            
            
    





    def generate_training_execution_constrained_failures(self,steps,lookahead = 1,lookahead_function = None,environment_strategy = None,print_probs = False,random_exploration = False,random_step = 4,new_loss = False,epsilon = 0,average_failures = 0,min_failures = 0,max_failures = 0):
        
        
        
        #same as before but the loss in computed according the the constrained 
        #number of failures min_failures and max_failures
        
        rollout = [None]*(lookahead+1)
        rollout_error = [None]*(lookahead+1)
        
        
        def rollout_update(rollout,new_state):
            return rollout[1:]+[new_state]
            
        def rollout_error_update(rollout_errors,error):
            return rollout_errors[1:]+[error]

        if self.model == "any_available":
            execution = []
            state = self.network.initial_state()
            last_transition = None
            output = None
            output_value = None
            
            past_loss = 0
            nb_fail = 0
            
            dy.renew_cg()
            state = self.network.initial_state()
            
            
            loss = [dy.scalarInput(0)]
            
            
            for step in range(steps):
                
                network_input = self.RNN_input(last_transition)

                input_vector = dy.inputVector(network_input)
                state = state.add_input(input_vector)
                output = dy.softmax(self.R*state.output() + self.bias)
                
                
                
                output_value = output.value()
                
                if not random_exploration or step > random_step:
                    if random.random() < epsilon:
                        next_plant_action = self.random_transition()[0]
                    else:
                        next_plant_action = self.RNN_output(output_value,print_probs = print_probs)
                else:
                    next_plant_action = random.choice(self.plant.available_transitions())
                
                if environment_strategy == None:
                    environment_action = random.choice([tr for tr in self.environment.internal + self.environment.shared if tr.start == self.environment.current_state]).name
                else:
                    environment_action = environment_strategy(self.environment)
                    
                    
                tr = self.check_transition(next_plant_action,environment_action,environment_strategy)
                plant_tr = tr[0]
                if plant_tr[:4] == "fail":
                    plant_tr = plant_tr[5]
                    if not random_exploration or step > random_step:
                        rollout_error = rollout_error_update(rollout_error,True)
                else:
                    if not random_exploration or step > random_step:
                        rollout_error = rollout_error_update(rollout_error,False)
                    
                if not random_exploration or step > random_step:
                    rollout = rollout_update(rollout,(output,self.plant.available_transitions(),plant_tr))
                
                
                
                available = self.plant.available_transitions()
                
                if not random_exploration or step > random_step:
                
                    i_train = next(i for i in range(len(rollout)) if rollout[i] != None)
                    
                    if rollout[i_train] != None:
                        
                        
                        nb_failures = rollout_error.count(True)
                        nb_successes = lookahead-nb_failures
                        
# =============================================================================
#                         #between min and max
#                         
#                         for i in range(len(self.plant.all_transitions)):
#                             if self.plant.all_transitions[i] in rollout[i_train][1]:
#                                 if self.plant.all_transitions[i] == rollout[i_train][2]:#chosen action
#                                     if nb_failures <= max_failures and nb_failures >= min_failures:
#                                         loss.append((lookahead-max_failures)*dy.pickneglogsoftmax(rollout[i_train][0],i))
#                                         
#                                 # original loss
#                                 if not new_loss:
#                                     if nb_failures > max_failures or nb_failures < min_failures:
#                                         loss.append(dy.pickneglogsoftmax(rollout[i_train][0],i))
#                                 ###############
#                                 
#                                 # Yoav's correction of the loss
#                                 else:
#                                     if self.plant.all_transitions[i] != rollout[i_train][2]:#not chosen action
#                                         if nb_failures > max_failures or nb_failures < min_failures:
#                                             loss.append((max_failures)*dy.pickneglogsoftmax(rollout[i_train][0],i))
#                                 ###############
# 
# =============================================================================
                        
                        
                        #close to average
                        
                        for i in range(len(self.plant.all_transitions)):
                            if self.plant.all_transitions[i] in rollout[i_train][1]:
                                if self.plant.all_transitions[i] == rollout[i_train][2]:#chosen action
                                    loss.append((lookahead-(average_failures - nb_failures))*dy.pickneglogsoftmax(rollout[i_train][0],i))
                                        
                                # original loss
                                if not new_loss:
                                    loss.append((average_failures - nb_failures)*dy.pickneglogsoftmax(rollout[i_train][0],i))
                                ###############
                                
                                # Yoav's correction of the loss
                                else:
                                    if self.plant.all_transitions[i] != rollout[i_train][2]:#not chosen action
                                        loss.append((average_failures - nb_failures)*dy.pickneglogsoftmax(rollout[i_train][0],i))
                                ###############

                        
                    #print("training over",execution,"with",nb_failures,"failures")
                    loss_compute = dy.esum(loss) #* (1+nb_fail)
    
        
                    loss_compute.value()
                    #print(loss_compute.value())
                    loss_compute.backward()
                    self.trainer.update()
                    
                    loss = [dy.scalarInput(0)]    
                    
                if self.plant.current_state in self.plant.update_states:
                    
                    pass
                    nb_fail = 0
                    
                    #loss = [dy.scalarInput(0)]
                    
                    
                self.trigger_transition(tr)
                #print(tr)
                if tr[0][:4] == "fail":
                    nb_fail += 1
                    past_loss = past_loss +1
# =============================================================================
#                 else:
#                     nb_fail = 0
# =============================================================================

                execution.append(tr)
                last_transition = tr[0]
            return(execution)
            
            
            
# =============================================================================
#     def default_lookahead(self,history):
#         return 0
# =============================================================================        
            
            
        
        
        
        
