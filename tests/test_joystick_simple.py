import pygame


def test_f310():
    pygame.init()

    # ジョイスティックの初期化
    pygame.joystick.init()

    joystick_count = pygame.joystick.get_count()
    print(f"ジョイスティックの数: {joystick_count}")

    if joystick_count > 0:
        joystick = pygame.joystick.Joystick(1)  # ASROCKのLEDコントローラが0番目に認識されていた
        joystick.init()
        print(f"ジョイスティックの名前: {joystick.get_name()}")
        print(f"軸の数: {joystick.get_numaxes()}")
        print(f"ボタンの数: {joystick.get_numbuttons()}")

        while True:
            for event in pygame.event.get():
                if event.type == pygame.JOYAXISMOTION:
                    print(f"軸の動き: {event.axis}, 値: {event.value}")
                elif event.type == pygame.JOYBUTTONDOWN:
                    print(f"ボタンが押されました: {event.button}")
                elif event.type == pygame.JOYBUTTONUP:
                    print(f"ボタンが離されました: {event.button}")
    else:
        print("F310が接続されていません。")


if __name__ == "__main__":
    test_f310()
