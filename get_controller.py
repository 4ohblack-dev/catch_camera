import time
import pygame


def main():
    pygame.init()
    pygame.joystick.init()

    controller_count = pygame.joystick.get_count()

    if controller_count == 0:
        print("コントローラが見つかりません。")
        pygame.quit()
        return

    joy = pygame.joystick.Joystick(0)
    joy.init()

    print("コントローラ名:", joy.get_name())
    print("軸の数:", joy.get_numaxes())
    print("ボタン数:", joy.get_numbuttons())
    print("Hat数:", joy.get_numhats())
    print()
    print("十字キーやスティック、ボタンを操作してください。")
    print("終了するには Ctrl + C を押してください。")

    try:
        while True:
            # コントローラの最新状態を取得
            pygame.event.pump()

            axes = [
                round(joy.get_axis(i), 2)
                for i in range(joy.get_numaxes())
            ]

            pressed_buttons = [
                i
                for i in range(joy.get_numbuttons())
                if joy.get_button(i)
            ]

            hats = [
                joy.get_hat(i)
                for i in range(joy.get_numhats())
            ]

            print(
                f"axes={axes}  "
                f"buttons={pressed_buttons}  "
                f"hats={hats}"
            )

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n終了します。")

    finally:
        joy.quit()
        pygame.quit()


if __name__ == "__main__":
    main()