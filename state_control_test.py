import time

import cv2
import numpy as np
import pygame
from ultralytics import YOLO


SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
CENTER_X = SCREEN_WIDTH / 2
CENTER_Y = SCREEN_HEIGHT / 2

# controller_check.py で調べたボタン番号を設定する
DPAD_LEFT = 13
DPAD_RIGHT = 14
DPAD_UP = 11
DPAD_DOWN = 12


def deadzone(value, threshold=0.05):
    return 0.0 if abs(value) < threshold else value


def get_axis(joy, index):
    """存在しない軸を読もうとしても停止しないようにする。"""
    if index < joy.get_numaxes():
        return deadzone(joy.get_axis(index))
    return 0.0


def get_button(joy, index):
    """存在しないボタンを読もうとしても停止しないようにする。"""
    if index < joy.get_numbuttons():
        return int(joy.get_button(index))
    return 0


def main():
    # --- YOLO ---
    model = YOLO("best_seg_openvino_model", task="segment")

    # --- Camera ---
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

    if not cap.isOpened():
        print("カメラを開けませんでした。")
        return

    # --- Controller ---
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("コントローラが見つかりません。")
        cap.release()
        pygame.quit()
        return

    joy = pygame.joystick.Joystick(0)
    joy.init()

    print("コントローラ名:", joy.get_name())
    print("軸の数:", joy.get_numaxes())
    print("ボタン数:", joy.get_numbuttons())
    print("Hat数:", joy.get_numhats())
    print("開始しました。カメラ画面で q または Esc を押すと終了します。")

    last_print_time = 0.0

    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("カメラ映像を取得できませんでした。")
                break

            pygame.event.pump()

            # 初期値：検出なし
            delta_x = 0.0
            delta_y = 0.0
            angle = 0.0
            state = 0
            best_contour = None

            # --- YOLOセグメンテーション ---
            results = model.predict(
                frame,
                stream=False,
                verbose=False,
                conf=0.6,
                retina_masks=True
            )

            result = results[0]

            if result.masks is not None:
                masks_data = result.masks.data.cpu().numpy()

                # 全マスクの輪郭から、最大の輪郭を選ぶ
                for mask in masks_data:
                    mask_image = (mask * 255).astype(np.uint8)

                    contours, _ = cv2.findContours(
                        mask_image,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )

                    if contours:
                        contour = max(contours, key=cv2.contourArea)

                        if (
                            best_contour is None
                            or cv2.contourArea(contour) > cv2.contourArea(best_contour)
                        ):
                            best_contour = contour

            # --- 最大物体の位置・角度と〇/✕判定 ---
            if best_contour is not None:
                hull = cv2.convexHull(best_contour)
                rect = cv2.minAreaRect(hull)

                center_x, center_y = rect[0]
                width, height = rect[1]
                angle = rect[2]

                if width < height:
                    angle += 90

                angle = angle % 180
                delta_x = center_x - CENTER_X
                delta_y = center_y - CENTER_Y

                # 中心から100px以内なら〇
                state = int(delta_x ** 2 + delta_y ** 2 < 100 ** 2)

                box = cv2.boxPoints(rect)
                box = np.intp(box)
                cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
                cv2.circle(frame, (int(center_x), int(center_y)), 6, (0, 0, 255), -1)

            # --- コントローラ入力 ---
            left_y = get_axis(joy, 1)
            right_y = get_axis(joy, 3)

            l_button = get_button(joy, 9)
            r_button = get_button(joy, 10)

            button_left = get_button(joy, DPAD_LEFT)
            button_right = get_button(joy, DPAD_RIGHT)
            button_up = get_button(joy, DPAD_UP)
            button_down = get_button(joy, DPAD_DOWN)

            # --- 画面表示 ---
            symbol = "OK" if state else "NG"
            color = (0, 255, 0) if state else (0, 0, 255)

            cv2.putText(
                frame,
                f"Target: {symbol}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                2
            )

            cv2.putText(
                frame,
                f"dx={delta_x:.1f}, dy={delta_y:.1f}, angle={angle:.1f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.circle(
                frame,
                (int(CENTER_X), int(CENTER_Y)),
                100,
                color,
                2
            )

            cv2.imshow("YOLO / Controller check", frame)

            # ターミナルには毎秒10回だけ表示（ログが流れすぎないようにする）
            now = time.monotonic()
            if now - last_print_time >= 0.1:
                mark = "〇" if state else "✕"

                print(
                    f"判定={mark} "
                    f"dx={delta_x:+.1f}, dy={delta_y:+.1f}, angle={angle:.1f} | "
                    f"leftY={left_y:+.2f}, rightY={right_y:+.2f} | "
                    f"L={l_button}, R={r_button} | "
                    f"D-pad: 左={button_left}, 右={button_right}, "
                    f"上={button_up}, 下={button_down}"
                )

                last_print_time = now

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        joy.quit()
        pygame.quit()


if __name__ == "__main__":
    main()