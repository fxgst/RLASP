# set test parameters for generating test data
# results will be saved in /testdata

num_blocks 				= 5 # just for the test name, need to add/remove blocks and subgoals in .lp files under "input predicates"!
number_episodes 		= 400
number_runs 			= 10
planning_factor     	= 0.1
plan_on_empty_policy	= False
planning_horizon		= 2

max_episode_len			= num_blocks*2
mode 					= 'mean'
plot_points 			= 200
