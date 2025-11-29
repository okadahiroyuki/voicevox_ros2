import os
import io
import re

import numpy as np
import sounddevice as sd
import soundfile as sf

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from voicevox_core.blocking import Onnxruntime, OpenJtalk, Synthesizer, VoiceModelFile


SPEAKER_PREFIX_RE = re.compile(r"^\s*\[(\d+)\]\s*(.*)$")


def find_onnxruntime_lib(engine_dir: str, logger) -> str:
    """
    engine_dir 以下を再帰的に探索して
    Onnxruntime.LIB_VERSIONED_FILENAME に対応するファイルを探す。
    見つかったパスを返す。見つからなければ例外。
    """
    target_name = Onnxruntime.LIB_VERSIONED_FILENAME
    logger.info(f"Searching for ONNX Runtime library '{target_name}' under {engine_dir} ...")

    for root, dirs, files in os.walk(engine_dir):
        if target_name in files:
            path = os.path.join(root, target_name)
            logger.info(f"Found ONNX Runtime library at: {path}")
            return path

    raise FileNotFoundError(f"ONNX Runtime library '{target_name}' not found under {engine_dir}")

def find_openjtalk_dict_dir(engine_dir: str, logger) -> str:
    """
    engine_dir 以下を再帰的に探索して
    open_jtalk_dic_utf_8-1.11 ディレクトリを探す。
    見つかったパスを返す。
    """
    target_dir_name = "open_jtalk_dic_utf_8-1.11"
    logger.info(f"Searching for OpenJTalk dict '{target_dir_name}' under {engine_dir} ...")

    for root, dirs, files in os.walk(engine_dir):
        if target_dir_name in dirs:
            path = os.path.join(root, target_dir_name)
            logger.info(f"Found OpenJTalk dict at: {path}")
            return path

    raise FileNotFoundError(
        f"OpenJTalk dict '{target_dir_name}' not found under {engine_dir}"
    )

def find_vvm_path(engine_dir: str, vvm_file: str, logger) -> str:
    """
    engine_dir 以下を再帰的に探索して
    指定された vvm ファイル (例: "0.vvm") を探す。
    見つかったフルパスを返す。
    """
    logger.info(f"Searching for VVM file '{vvm_file}' under {engine_dir} ...")

    for root, dirs, files in os.walk(engine_dir):
        if vvm_file in files:
            path = os.path.join(root, vvm_file)
            logger.info(f"Found VVM file at: {path}")
            return path

    raise FileNotFoundError(f"VVM file '{vvm_file}' not found under {engine_dir}")


class VoicevoxTTSNode(Node):
    def __init__(self):
        super().__init__("voicevox_tts_node")

        # --- パラメータ ---
        self.declare_parameter(
        #    "engine_dir", "/home/roboworks/voicevox_engine"
            "engine_dir", "/voicevox_engine"
        )
        # 使用する vvm ファイル（必要に応じて変えてください）
        self.declare_parameter("vvm_file", "0.vvm")
        # デフォルトの style_id（メッセージに指定がなければこれを使う）
        self.declare_parameter("style_id", 0)

        engine_dir = self.get_parameter("engine_dir").get_parameter_value().string_value
        vvm_file = self.get_parameter("vvm_file").get_parameter_value().string_value
        self.default_style_id = (
            self.get_parameter("style_id").get_parameter_value().integer_value
        )

        self.get_logger().info(f"engine_dir: {engine_dir}")
        self.get_logger().info(f"vvm_file : {vvm_file}")
        self.get_logger().info(f"default style_id : {self.default_style_id}")

        # --- VOICEVOX Core の初期化 ---
        try:
            #onnxruntime_path = os.path.join(
            #    engine_dir, "onnxruntime", "lib", Onnxruntime.LIB_VERSIONED_FILENAME
            #)
            # 修正版: engine_dir 以下を探索して見つける
            onnxruntime_path = find_onnxruntime_lib(engine_dir, self.get_logger())	
            #open_jtalk_dict_dir = os.path.join(
            #    engine_dir, "dict", "open_jtalk_dic_utf_8-1.11"
            #)
            open_jtalk_dict_dir = find_openjtalk_dict_dir(engine_dir, self.get_logger())
            #vvm_path = os.path.join(engine_dir, "models", "vvms", vvm_file)
            vvm_path = find_vvm_path(engine_dir, vvm_file, self.get_logger())

            self.get_logger().info(f"onnxruntime: {onnxruntime_path}")
            self.get_logger().info(f"open_jtalk : {open_jtalk_dict_dir}")
            self.get_logger().info(f"vvm       : {vvm_path}")

            # 1. Synthesizer 初期化
            onnx = Onnxruntime.load_once(filename=onnxruntime_path)
            openjtalk = OpenJtalk(open_jtalk_dict_dir)
            self.synthesizer = Synthesizer(onnx, openjtalk)

            # 2. 音声モデル読み込み
            with VoiceModelFile.open(vvm_path) as model:
                self.synthesizer.load_voice_model(model)

            self.get_logger().info("VOICEVOX Synthesizer 初期化完了")

        except Exception as e:
            self.get_logger().error(f"VOICEVOX 初期化に失敗しました: {e}")
            raise

        # --- 購読設定 ---
        self.subscription = self.create_subscription(
            String, "tts_text", self.on_text_received, 10
        )

    def parse_speaker_and_text(self, raw_text: str):
        """
        先頭に [数字] があればその数字を style_id として使い、
        残りをテキストとして返す。
        例: "[1] こんにちは" -> (1, "こんにちは")
            "普通のテキスト" -> (self.default_style_id, "普通のテキスト")
        """
        m = SPEAKER_PREFIX_RE.match(raw_text)
        if m:
            style_id = int(m.group(1))
            text = m.group(2)
            return style_id, text
        else:
            return self.default_style_id, raw_text

    def on_text_received(self, msg: String):
        if not msg.data:
            return

        style_id, text = self.parse_speaker_and_text(msg.data)

        if not text:
            self.get_logger().warn("空文字列が来たのでスキップします")
            return

        self.get_logger().info(f"TTS(style_id={style_id}): 「{text}」")

        try:
            # テキストから WAV (bytes) 生成
            wav_bytes = self.synthesizer.tts(text, style_id)

            # バッファから直接再生
            data, samplerate = sf.read(io.BytesIO(wav_bytes), dtype="int16")
            data = np.asarray(data)

            sd.play(data, samplerate)
            sd.wait()

        except Exception as e:
            self.get_logger().error(f"TTS 再生中にエラー: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = VoicevoxTTSNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

