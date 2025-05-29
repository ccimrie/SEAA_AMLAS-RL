import numpy as np
import pickle

dtmc_file_template_dir='dtmc_template'

dtmc_model=open(f'{dtmc_file_template_dir}/dtmc_start.txt', 'r').readlines()

file = open("data_collection/markov_model.pickle",'rb')
object_file = pickle.load(file)
combinations=object_file.transition_matrix_dict
initialisation_states=object_file.initial_state

total_sum=np.sum([initialisation_states[i_s] for i_s in initialisation_states])
initialisation_line_start='[] t=-1 ->\n'
initialisation_state_template='{0}:(vehicle_stage\'={1})&(u_dist\'={2})&(o_dist\'={3})&(t\'=0)'
for i_s in initialisation_states:
	p=initialisation_states[i_s]/total_sum
	line=initialisation_state_template.format(p, i_s[0], i_s[1], i_s[2])
	initialisation_line_start+=(line+' + \n')

state_transition_lines=initialisation_line_start[:-3]+';\n\n'
for combination in combinations:
	if len(combinations[combination])>0:
		com_transitions=combinations[combination]
		total_sum=np.sum([com_transitions[next_state] for next_state in com_transitions])
		transition_lines=f"[] t>-1 & t<e_lim & vehicle_stage={combination[0]} & u_dist={combination[1]} & o_dist={combination[2]} -> \n"
		transition_template="{0}: (vehicle_stage'={1})&(u_dist'={2})&(o_dist'={3})&(t'=t+1)"
		for next_state in com_transitions:
			p=com_transitions[next_state]/total_sum
			transition_line=transition_template.format(p, next_state[0], next_state[1], next_state[2])
			transition_lines+=(transition_line+' +\n')
		state_transition_lines+=transition_lines[:-3]+';\n\n'

dtmc_model_strip=[line.strip() for line in dtmc_model]
ind=dtmc_model_strip.index('endmodule')

dtmc_model.insert(ind, state_transition_lines)

with open('../dtmc_model_checking/rl_vehicle_dtmc.pm', 'w') as f:
	[f.write(line) for line in dtmc_model]
