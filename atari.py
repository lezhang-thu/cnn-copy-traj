import numpy as np
from collections import deque
import gymnasium as gym
import cv2
import ale_py

gym.register_envs(ale_py)


class LazyFrames(object):

    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]


class Atari(gym.Env):
    WEIGHTS = np.array([0.299, 0.587, 1 - (0.299 + 0.587)])

    def __init__(
        self,
        name,
        action_repeat=4,
        #size=(96, 96),
        size=(84, 84),
        gray=True,
        noops=0,
        sticky=False,
        actions="all",
        #length=108_000,
        length=20_000,
    ):
        #def __init__(
        #    self,
        #    name,
        #    action_repeat=4,
        #    size=(84, 84),
        #    gray=True,
        #    noops=0,
        #    lives="unused",
        #    sticky=False,
        #    actions="all",
        #    length=20_000,
        #    resize="opencv",
        #    seed=None,
        #):

        self._cv2 = cv2
        self._repeat = action_repeat
        assert self._repeat == 4
        self._size = size
        self._gray = gray

        print("self._size: {}".format(self._size))
        print("self._gray : {}".format(self._gray))
        self._noops = noops
        self._sticky = sticky
        self._length = length
        self._env = gym.make(
            'ALE/{}-v5'.format(name),
            obs_type='rgb',  # ram | rgb | grayscale
            frameskip=1,  # frame skip
            mode=None,  # game mode, see Machado et al. 2018
            difficulty=None,  # game difficulty, see Machado et al. 2018
            repeat_action_probability=0.25
            if sticky else 0.0,  # Sticky action probability
            full_action_space=actions == "all",  # Use all actions
            render_mode=None  # None | human | rgb_array
        )
        assert self._env.unwrapped.get_action_meanings()[0] == "NOOP"
        shape = self._env.observation_space.shape
        self._buffer = [np.zeros(shape, np.uint8) for _ in range(2)]
        self._ale = self._env.unwrapped.ale

        self.frame_stack = 4
        self.frames = deque([], maxlen=self.frame_stack)

        self._done = None
        self._step = 0

    def _get_ob(self):
        assert len(self.frames) == self.frame_stack
        return LazyFrames(list(self.frames))

    @property
    def observation_space(self):
        img_shape = self._size + ((1 * self.frame_stack, ) if self._gray else
                                  (3 * self.frame_stack, ))
        return gym.spaces.Dict({
            "image":
            gym.spaces.Box(0, 255, img_shape, np.uint8),
        })

    @property
    def action_space(self):
        space = self._env.action_space
        space.discrete = True
        return space

    def step(self, action):
        acc_rew = 0.0
        for repeat in range(self._repeat):
            _, reward, over, _, info = self._env.step(action)
            self._step += 1
            acc_rew += reward
            if repeat == self._repeat - 2:
                self._screen(self._buffer[1])
            if over:
                break
        self._screen(self._buffer[0])
        self._done = over or (self._length and self._step >= self._length)
        ob = self._obs()
        self.frames.append(ob)
        return self._get_ob(), acc_rew, self._done

    def reset(self):
        self._env.reset(seed=0)

        self._screen(self._buffer[0])
        # numpy.copyto(dst, src, casting='same_kind', where=True)
        np.copyto(self._buffer[1], self._buffer[0])
        self._step = 0
        self._done = False
        ob = self._obs()
        for _ in range(self.frame_stack):
            self.frames.append(ob)
        return self._get_ob()

    def _obs(self):
        np.maximum(self._buffer[0], self._buffer[1], out=self._buffer[0])
        image = self._buffer[0]
        if image.shape[:2] != self._size:
            image = self._cv2.resize(image,
                                     self._size,
                                     interpolation=self._cv2.INTER_AREA)
        if self._gray:
            image = (image * self.WEIGHTS).sum(-1).astype(image.dtype)[:, :,
                                                                       None]
        return image

    def _screen(self, array):
        self._ale.getScreenRGB(array)

    def close(self):
        return self._env.close()
