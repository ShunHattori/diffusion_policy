import multiprocessing as mp
import time

import numpy as np
import pygame

from diffusion_policy.shared_memory.shared_memory_ring_buffer import (
    SharedMemoryRingBuffer,
)


class JoyStick(mp.Process):
    def __init__(
        self,
        shm_manager,
        get_max_k=30,
        frequency=200,
        max_value=32767,  # pygameのJoy-Stick軸値は -32768 ~ 32767
        deadzone=(0.06, 0.06, 0.06, 0.06, 0.06, 0.06),
        dtype=np.float32,
        n_buttons=12,
    ):
        """
        Continuously listen to Joy-Stick events and update the latest state.

        max_value: Maximum value of axis data (pygame joystick ranges from -32768 to 32767)
        deadzone: [0,1], number or tuple, axis with value lower than this value will stay at 0

        Coordinate system for Joy-Stick:
        x-axis: left/right
        y-axis: up/down
        z-axis: rotation
        """
        super().__init__()
        if np.issubdtype(type(deadzone), np.number):
            deadzone = np.full(6, fill_value=deadzone, dtype=dtype)
        else:
            deadzone = np.array(deadzone, dtype=dtype)
        assert (deadzone >= 0).all()

        self.reverse_axis = [False, True, False, False]

        # copied variables
        self.frequency = frequency
        self.max_value = max_value
        self.dtype = dtype
        self.deadzone = deadzone
        self.n_buttons = n_buttons

        example = {
            "motion_event": np.zeros((7,), dtype=np.int64),  # 3 translation, 3 rotation, 1 period
            "button_state": np.zeros((n_buttons,), dtype=bool),
            "receive_timestamp": time.time(),
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency,
        )

        # shared variables
        self.ready_event = mp.Event()
        self.stop_event = mp.Event()
        self.ring_buffer = ring_buffer

    # ======= get state APIs ==========

    def get_motion_state(self):
        state = self.ring_buffer.get()
        state = np.array(state["motion_event"][:6], dtype=self.dtype) / self.max_value
        is_dead = (-self.deadzone < state) & (state < self.deadzone)
        state[is_dead] = 0
        return state

    def get_motion_state_transformed(self):
        """
        Return in Joy-Stick's coordinate system.
        """
        return self.get_motion_state()

    def get_button_state(self):
        state = self.ring_buffer.get()
        return state["button_state"]

    def is_button_pressed(self, button_id):
        return self.get_button_state()[button_id]

    # ========== start stop API ===========

    def start(self, wait=True):
        super().start()
        if wait:
            self.ready_event.wait()

    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= main loop ==========
    def run(self):
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No Joy-Stick connected!")
        joystick = pygame.joystick.Joystick(1)  # ASROCKのLEDコントローラが0番目に認識されていたため1番目を選択
        joystick.init()
        print(f"Start listening to Joy-Stick: {joystick.get_name()}")

        try:
            motion_event = np.zeros((7,), dtype=np.int64)
            button_state = np.zeros((self.n_buttons,), dtype=bool)
            # send one message immediately so client can start reading
            self.ring_buffer.put(
                {
                    "motion_event": motion_event,
                    "button_state": button_state,
                    "receive_timestamp": time.time(),
                },
            )
            self.ready_event.set()

            while not self.stop_event.is_set():
                # pygameイベント処理
                for event in pygame.event.get():
                    receive_timestamp = time.time()
                    if event.type == pygame.JOYAXISMOTION:
                        # 軸の動きをmotion_eventに保存 (軸は0~5まで対応）
                        if event.axis < 3:
                            if self.reverse_axis[event.axis]:
                                motion_event[event.axis] = -int(event.value * self.max_value)  # translation
                            else:
                                motion_event[event.axis] = int(event.value * self.max_value)  # translation
                        else:
                            motion_event[event.axis] = int(event.value * self.max_value)  # rotation

                    elif event.type == pygame.JOYBUTTONDOWN:
                        button_state[event.button] = True

                    elif event.type == pygame.JOYBUTTONUP:
                        button_state[event.button] = False

                # データをリングバッファに送信
                self.ring_buffer.put(
                    {
                        "motion_event": motion_event,
                        "button_state": button_state,
                        "receive_timestamp": receive_timestamp,
                    }
                )

                time.sleep(1 / self.frequency)

        finally:
            pygame.quit()
