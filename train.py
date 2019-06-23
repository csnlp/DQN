import random
import math
from itertools import count

import gym
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from DQN import DQN
from utils import get_screen
from utils import plot_durations
from memory import ReplayMemory
from memory import Transition

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-steps_done / float(EPS_DECAY))
    steps_done += 1

    # random strategy: at begining always take the random strategy
    if sample < eps_threshold:
        return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)
    else:
        return policy_net(state).max(1)[1].view(1,1)


def optimize_model(policy_net, optimizer):
    # first sample a batch
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    # non_final_mask is the mask to tag all the item whose next_state is not None as True
    non_final_mask = tuple(map(lambda s: s is not None, batch.next_state))
    non_final_mask = torch.tensor(non_final_mask, device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # policy_net(state_batch) is used to get all value among all actions
    # gather method is used to get the value corresponding to certain action
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    # compute the V(s_{t+1}) for $s_{t+1}$ which is final state, we set V(s_{t+1}) = 0
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

env = gym.make('CartPole-v0').unwrapped
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env.reset()

BATCH_SIZE = 128
# GAMMA is the discount factor
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

TARGET_UPDATE = 10

AVERAGE_SIZE = 10
episode_durations = []

init_screen = get_screen(env, device)
_, _, screen_height, screen_width = init_screen.shape

policy_net = DQN(screen_height, screen_width).to(device)
target_net = DQN(screen_height, screen_width).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0
num_episodes = 300
for i_episode in range(num_episodes):
    env.reset()
    last_screen = get_screen(env, device)
    current_screen = get_screen(env, device)
    state = current_screen - last_screen
    #print state
    for t in count():
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        last_screen = current_screen
        current_screen = get_screen(env, device)

        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        memory.push(state, action, next_state, reward)
        
        state = next_state
        #if done:
        #    print "Episode Done"
        #else:
        #    print state.size()
        optimize_model(policy_net, optimizer)
        if done:
            episode_durations.append(t+1)
            plot_durations(episode_durations, AVERAGE_SIZE)
            break

    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print("Complet")
env.render()
env.close()
plt.ioff()
plt.show()



