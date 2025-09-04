import dearpygui.dearpygui as dpg
import os
from faster_whisper import WhisperModel
import threading
import json
from datetime import datetime


class TranscriptionService:
    @staticmethod
    def transcribe(audio_path, language=None):
        model = WhisperModel("large-v3", device="cpu", compute_type="int8")
        if language == "auto":
            language = None
        segments, info = model.transcribe(audio_path, language=language)
        detected_language = info.language
        segments = list(segments)
        return detected_language, segments


class WhisperGUI:
    def __init__(self):
        self.selected_file = ""
        self.is_transcribing = False
        self.last_result = ""

    def file_selector_callback(self, sender, app_data):
        self.selected_file = list(app_data["selections"].values())[0]
        dpg.set_value(
            "file_path", f"選択されたファイル: {os.path.basename(self.selected_file)}"
        )
        dpg.configure_item("transcribe_button", enabled=True)

    def transcribe_audio(self):
        if not self.selected_file or self.is_transcribing:
            return

        self.is_transcribing = True
        dpg.configure_item("transcribe_button", enabled=False, label="処理中...")
        dpg.set_value("result_text", "音声ファイルを処理しています...")

        def transcribe_thread():
            try:
                selected_language_display = dpg.get_value("language_combo")
                selected_language = self.language_mapping[selected_language_display]
                detected_language, segments = TranscriptionService.transcribe(
                    self.selected_file, selected_language
                )

                result = f"検出言語: {detected_language}\n\n"
                for segment in segments:
                    result += (
                        f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n"
                    )

                self.last_result = result
                dpg.set_value("result_text", result)
                dpg.configure_item("save_button", enabled=True)
                dpg.configure_item("copy_button", enabled=True)

            except Exception as e:
                dpg.set_value("result_text", f"エラーが発生しました: {str(e)}")

            finally:
                self.is_transcribing = False
                dpg.configure_item(
                    "transcribe_button", enabled=True, label="文字起こし実行"
                )

        threading.Thread(target=transcribe_thread, daemon=True).start()

    def save_result(self):
        if not self.last_result:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transcription_{timestamp}.txt"

        with dpg.file_dialog(
            directory_selector=False,
            show=True,
            callback=self.save_callback,
            tag="save_dialog",
            width=700,
            height=400,
            default_filename=filename,
        ):
            dpg.add_file_extension(".txt", color=(255, 255, 0, 255))

    def save_callback(self, sender, app_data):
        try:
            if "file_path_name" in app_data:
                file_path = app_data["file_path_name"]
            elif "selections" in app_data and app_data["selections"]:
                file_path = list(app_data["selections"].values())[0]
            else:
                dpg.set_value("status_text", "保存エラー: ファイルが選択されていません")
                return
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(self.last_result)
            dpg.set_value("status_text", f"保存完了: {os.path.basename(file_path)}")
        except Exception as e:
            dpg.set_value("status_text", f"保存エラー: {str(e)}")

    def copy_result(self):
        if not self.last_result:
            return
        dpg.set_clipboard_text(self.last_result)
        dpg.set_value("status_text", "クリップボードにコピーしました")
    
    def show_help(self):
        if dpg.does_item_exist("help_window"):
            dpg.show_item("help_window")
            return
        
        with dpg.window(
            label="ヘルプ - モデルダウンロードについて",
            tag="help_window",
            width=500,
            height=400,
            modal=True
        ):
            dpg.add_text("Whisper モデルの自動ダウンロードについて", color=(70, 130, 180))
            dpg.add_separator()
            dpg.add_spacer(height=10)
            
            help_text = """初回起動時のモデルダウンロード:

• アプリケーションを初めて実行すると、Whisperモデル（large-v3）が
  自動的にダウンロードされます

• ダウンロード元: Hugging Face Hub (Systranオーガニゼーション)

• モデルサイズ: 約1.5GB

• 保存先: システムのキャッシュディレクトリ
  (通常は ~/.cache/huggingface/)

• インターネット接続が必要です

• ダウンロード時間はネットワーク速度に依存します

• 2回目以降はキャッシュから高速で読み込まれます

注意事項:
- 十分なディスク容量を確保してください
- 初回実行時は数分かかる場合があります
- ダウンロード中は「処理中...」と表示されます"""

            dpg.add_input_text(
                default_value=help_text,
                multiline=True,
                readonly=True,
                width=-1,
                height=250
            )
            
            dpg.add_spacer(height=10)
            with dpg.group(horizontal=True):
                dpg.add_spacer()
                dpg.add_button(
                    label="閉じる",
                    callback=lambda: dpg.hide_item("help_window"),
                    width=100
                )
                dpg.add_spacer()

    def create_gui(self):
        dpg.create_context()

        # フォント設定
        font_paths = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/Arial Unicode.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        ]

        font_loaded = False
        with dpg.font_registry():
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        with dpg.font(font_path, 18) as font:
                            dpg.add_font_range_hint(dpg.mvFontRangeHint_Japanese)
                            dpg.add_font_range_hint(
                                dpg.mvFontRangeHint_Chinese_Simplified_Common
                            )
                            dpg.add_font_range_hint(dpg.mvFontRangeHint_Chinese_Full)
                            dpg.add_font_range_hint(dpg.mvFontRangeHint_Cyrillic)
                            dpg.add_font_range_hint(dpg.mvFontRangeHint_Korean)
                        dpg.bind_font(font)
                        font_loaded = True
                        break
                    except:
                        continue

        if not font_loaded:
            print(
                "警告: CJKフォントが見つかりませんでした。文字化けする可能性があります。"
            )

        # モダンテーマの設定
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(
                    dpg.mvThemeCol_WindowBg,
                    (25, 25, 35, 255),
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_ChildBg,
                    (35, 35, 45, 255),
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_FrameBg,
                    (45, 45, 55, 255),
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_FrameBgHovered,
                    (55, 55, 65, 255),
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_FrameBgActive,
                    (65, 65, 75, 255),
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_Button,
                    (70, 130, 180, 255),
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_ButtonHovered,
                    (90, 150, 200, 255),
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_ButtonActive,
                    (110, 170, 220, 255),
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_Text,
                    (220, 220, 220, 255),
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_TitleBg,
                    (35, 35, 45, 255),
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_TitleBgActive,
                    (45, 45, 55, 255),
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_Header,
                    (70, 130, 180, 80),
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_HeaderHovered,
                    (90, 150, 200, 120),
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_HeaderActive,
                    (110, 170, 220, 160),
                    category=dpg.mvThemeCat_Core,
                )

                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowRounding, 8, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FrameRounding, 5, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 15, 15, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 8, 6, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_ItemSpacing, 8, 8, category=dpg.mvThemeCat_Core
                )

        dpg.bind_theme(global_theme)

        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self.file_selector_callback,
            tag="file_dialog",
            width=700,
            height=400,
        ):
            dpg.add_file_extension(".wav", color=(255, 255, 0, 255))
            dpg.add_file_extension(".mp3", color=(255, 255, 0, 255))
            dpg.add_file_extension(".m4a", color=(255, 255, 0, 255))

        with dpg.window(label="Whisper 文字起こしアプリ", tag="primary_window"):
            # ヘッダー部分
            dpg.add_text(
                "音声ファイルを選択して文字起こしを実行してください",
                color=(180, 180, 180),
            )
            dpg.add_separator()
            dpg.add_spacer(height=10)

            # 設定部分（横並び）
            with dpg.group(horizontal=True):
                # ファイル選択部分
                with dpg.group():
                    dpg.add_text("ファイル選択", color=(70, 130, 180))
                    dpg.add_button(
                        label="ファイルを選択",
                        callback=lambda: dpg.show_item("file_dialog"),
                    )

                dpg.add_spacer(width=30)

                # 言語選択部分
                with dpg.group():
                    dpg.add_text("言語設定", color=(70, 130, 180))
                    languages = [
                        ("自動検出", "auto"),
                        ("日本語", "ja"),
                        ("中国語", "zh"),
                        ("英語", "en"),
                        ("韓国語", "ko"),
                        ("フランス語", "fr"),
                        ("ドイツ語", "de"),
                        ("スペイン語", "es"),
                        ("イタリア語", "it"),
                        ("ロシア語", "ru"),
                    ]

                    dpg.add_combo(
                        items=[lang[0] for lang in languages],
                        default_value="自動検出",
                        tag="language_combo",
                        width=150,
                    )

                    # 言語コードのマッピングを保存
                    self.language_mapping = {lang[0]: lang[1] for lang in languages}

                dpg.add_spacer(width=30)

                # 実行ボタン部分
                with dpg.group():
                    dpg.add_text("", color=(70, 130, 180))  # 空白でレベル合わせ
                    dpg.add_button(
                        label="文字起こし実行",
                        callback=self.transcribe_audio,
                        tag="transcribe_button",
                        enabled=False,
                    )

            # ファイルパス表示
            dpg.add_spacer(height=15)
            dpg.add_text("", tag="file_path", color=(200, 200, 200), wrap=-1)

            dpg.add_separator()
            dpg.add_spacer(height=15)

            # 結果エリア
            with dpg.group(horizontal=True):
                dpg.add_text("結果:", color=(70, 130, 180))
                dpg.add_spacer()
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="ヘルプ",
                        callback=self.show_help,
                        tag="help_button",
                        width=60,
                        height=28,
                    )
                    dpg.add_button(
                        label="保存",
                        callback=self.save_result,
                        tag="save_button",
                        enabled=False,
                        width=60,
                        height=28,
                    )
                    dpg.add_button(
                        label="コピー",
                        callback=self.copy_result,
                        tag="copy_button",
                        enabled=False,
                        width=60,
                        height=28,
                    )

            dpg.add_spacer(height=10)
            dpg.add_input_text(
                tag="result_text",
                multiline=True,
                readonly=True,
                width=-1,
                height=320,
                default_value="ファイルを選択して実行してください...",
            )

            dpg.add_spacer(height=15)
            dpg.add_separator()
            dpg.add_text("Ready", tag="status_text", color=(100, 150, 100))

        dpg.create_viewport(title="Whisper GUI")
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("primary_window", True)
        dpg.start_dearpygui()
        dpg.destroy_context()


def main():
    app = WhisperGUI()
    app.create_gui()


if __name__ == "__main__":
    main()
