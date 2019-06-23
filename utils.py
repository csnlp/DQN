from PIL import Image

import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
import torchvision.transforms as T
import matplotlib.pyplot as plt


# resize is several transforms composed together
resize = T.Compose([
    T.ToPILImage(),
    T.Resize(40, interpolation=Image.CUBIC),
    T.ToTensor()
    ])

def plot_durations(episode_durations, AVERAGE_SIZE):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)

    plt.title('Training ...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    if len(durations_t) >= AVERAGE_SIZE:
        dim = 0
        size = AVERAGE_SIZE
        step = 1
        # duations_t.unfold(dim, size, step).size(): (no_point, 100)
        # duations_t.unfold(dim, size, step).mean(1).size(): (number_point, 1)
        means = durations_t.unfold(dim, size, step).mean(1).view(-1)
        means = torch.cat((torch.zeros(AVERAGE_SIZE-1), means))
        plt.plot(means.numpy())

    plt.pause(0.001)
    #if is_ipython:
    #    display.clear_output(wait=True)
    #    display.display(plt.gcf())

# Anyway, it is used to extract the abscissa asis of the cart
def get_cart_location(env, screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)

def get_screen(env, device):
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]

    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(env, screen_width)

    # slice usage: slice(stop) or slice(start, stop)
    # if in the left side
    if cart_location < view_width//2:
        slice_range = slice(view_width)
    # if in the right side
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    # if in the middle
    else:
        slice_range = slice(cart_location - view_width // 2, cart_location + view_width // 2)
    screen = screen[:, :, slice_range]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    
    screen = torch.from_numpy(screen)
    # add a batch dimension: BCHW
    return resize(screen).unsqueeze(0).to(device) 

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('CartPole-v0').unwrapped 
    env.reset()
    plt.figure()
    cart = get_screen(env).cpu().squeeze(0).permute(1, 2, 0).numpy()

    plt.imshow(cart, interpolation='none')
    plt.title('Cart')
    plt.show()
