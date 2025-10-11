# Question/main.py
import argparse  
import torch.multiprocessing as mp 
from dqn_agent_atari import AtariDQNAgent
from ddqn_agent_atari import AtariDDQNAgent
from dueling_dqn_agent_atari import AtariDuelingDQNAgent
from parallelized_dqn_agent import AtariParallelizedDQNAgent
# ```
# # run with the command: python main.py --agent type
# ```

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    parser = argparse.ArgumentParser(description="Train a agent on Atari games with different methods.")
    
    parser.add_argument(
        '--agent',        
        type=str,          
        required=True,      
        choices=['DQN', 'DDQN', 'DuelingDQN', 'parallelized'], 
        help="Specify the agent to use: DQN, DDQN, DuelingDQN, or parallelized" 
    )
    
    args = parser.parse_args()

    # my hyperparameters, you can change it as you like
    config = {
		"gpu": True,
		"training_steps": 2e7,
		"gamma": 0.99,
		"batch_size": 32,
		"eps_min": 0.1,
		"warmup_steps": 20000,
		"eps_decay": 1000000,
		"eval_epsilon": 0.01,
		"replay_buffer_capacity": 100000,
		"logdir": f'log/{args.agent}/MsPacman-v5/',
		"update_freq": 4,
		"update_target_freq": 10000,
		"learning_rate": 0.0000625,
        "eval_interval": 100,
        "eval_episode": 5,
		"env_id": 'ALE/MsPacman-v5',
        "num_actors": 4,
	}
    
    agent = None

    if args.agent == 'DQN':
        print(f"Initializing agent: DQN")
        print(f"Log directory: {config['logdir']}")
        agent = AtariDQNAgent(config)
    elif args.agent == 'DDQN':
        print(f"Initializing agent: DDQN")
        print(f"Log directory: {config['logdir']}")
        agent = AtariDDQNAgent(config)
    elif args.agent == 'DuelingDQN':
        print(f"Initializing agent: DuelingDQN")
        print(f"Log directory: {config['logdir']}")
        agent = AtariDuelingDQNAgent(config)
    elif args.agent == 'parallelized':
        print(f"Initializing agent: parallelized rollout")
        print(f"Log directory: {config['logdir']}")
        agent = AtariParallelizedDQNAgent(config)


    if agent:
        agent.train()
    else:
        print("Error: Invalid agent specified.")