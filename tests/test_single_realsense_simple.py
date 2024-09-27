import cv2
import numpy as np
import pyrealsense2 as rs


def main():
    # RealSenseパイプラインを作成
    pipeline = rs.pipeline()

    # コンフィギュレーションオブジェクトを作成してストリームを設定
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # カラーストリーム
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 深度ストリーム
    config.enable_device("135122075909")

    try:
        # パイプラインの開始
        pipeline.start(config)

        while True:
            # フレームを取得
            frames = pipeline.wait_for_frames()

            # カラーと深度フレームを取得
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # フレームデータをnumpy配列に変換
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # 深度画像を可視化するためにカラー画像として変換
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # カラー画像と深度画像を並べて表示
            images = np.hstack((color_image, depth_colormap))

            cv2.imshow("RealSense", images)

            # 'q'キーを押すと終了
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # パイプラインを停止
        pipeline.stop()


if __name__ == "__main__":
    main()
