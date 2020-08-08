# set test parameters for generating test data and plots
# results and blocks world will be saved in /testdata

num_blocks              = 4 # just for the test name, need to add/remove blocks and subgoals in *.lp files under "input predicates"
number_episodes         = 150
number_runs             = 2
planning_factor         = 0
plan_on_empty_policy    = False
planning_horizon        = 0 #2*(num_blocks-1)

max_episode_length      = num_blocks*3
mode                    = 'mean'
plot_points             = 100

