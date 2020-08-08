# set test parameters for generating test data and plots
# results and blocks world will be saved in /testdata

num_blocks              = 7 # just for the test name, need to add/remove blocks and subgoals in *.lp files under "input predicates"
number_episodes         = 5000
number_runs             = 20
planning_factor         = 0
plan_on_empty_policy    = False
planning_horizon        = 0 #2*(num_blocks-1)

max_episode_len         = num_blocks*3
mode                    = 'mean'
plot_points             = 250

