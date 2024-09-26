import time
from multiprocessing.managers import SharedMemoryManager

import numpy as np

from diffusion_policy.real_world.joystick_shared_memory import JoyStick


def test_joystick():
    # Joystickクラスのインスタンスを作成
    with SharedMemoryManager() as shm_manager:
        with JoyStick(
            shm_manager=shm_manager,  # シェアードメモリのマネージャー
            get_max_k=30,  # リングバッファのサイズ
            frequency=200,  # データ取得頻度 (Hz)
            max_value=32767,  # Joy-stick軸データの最大値
            deadzone=(0.001, 0.001, 0.001, 0.001, 0.001, 0.001),  # デッドゾーン設定
            dtype=np.float32,  # データ型
            n_buttons=12,  # ボタン数（Joy-stickは一般的に2ボタン）
        ) as joystick_process:
            try:
                print("Starting Joystick test. Move the sticks and press the buttons on your Joy-stick...")

                # 5秒間、Joystickのデータをリングバッファから読み取って表示する
                start_time = time.time()
                while time.time() - start_time < 5:
                    # 現在のモーションイベント（スティックの状態）を取得
                    motion_state = joystick_process.get_motion_state()
                    button_state = joystick_process.get_button_state()

                    print(f"Motion State: {motion_state}")
                    print(f"Button State: {button_state}")

                    state = joystick_process.is_button_pressed(7)
                    print(f"{state=}")

                    # 少し待機
                    time.sleep(0.1)

            finally:
                # Joystickプロセスを終了
                joystick_process.stop()
                print("Joystick test finished.")


if __name__ == "__main__":
    test_joystick()
