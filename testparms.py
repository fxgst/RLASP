# set test parameters for generating test data and plots
# results will be saved in /testdata

num_blocks              = 9 # just for the test name, need to add/remove blocks and subgoals in *.lp files under "input predicates"
number_episodes         = 2000
number_runs             = 4
planning_factor         = 0
plan_on_empty_policy    = True
planning_horizon        = num_blocks-1 #2*(num_blocks-1)

max_episode_len         = num_blocks*3
mode                    = 'mean'
plot_points             = 150
