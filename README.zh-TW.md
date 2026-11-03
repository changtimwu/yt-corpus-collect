# yt-corpus-collect

[English](README.md) | **繁體中文**

從 YouTube 收集語音與文字資料的工具，作為 ASR（自動語音辨識）與 TTS（文字轉語音）模型的訓練素材。

## 概覽

本專案使用 [yt-dlp](https://github.com/yt-dlp/yt-dlp) 從 YouTube 播放清單下載音訊與字幕，再打包成 [HuggingFace 相容](https://huggingface.co/docs/datasets)的 parquet 資料集——每個字幕段落一列，內嵌音訊，可直接用於微調。

## 安裝（macOS / Linux）

> [!NOTE]
> 以下步驟**僅適用於 macOS 與 Linux**。腳本與範例皆依賴 POSIX shell（`bash`/`zsh`、`nohup`、`source`、`peek.sh`）。Windows 未經測試，請改用 WSL2。

需求：Python 3.10+、ffmpeg、Node.js ≥ 20（yt-dlp 透過 `yt-dlp-ejs` 套件用它來破解 YouTube 的 n-sig 挑戰）。

**1. 安裝系統套件**

macOS（Homebrew）：

```bash
brew install python ffmpeg node
```

Linux（Debian/Ubuntu）：

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip ffmpeg
# 發行版內建的 nodejs 通常低於 20 版。請透過 NodeSource 或 nvm
# 安裝 Node.js ≥ 20，然後確認：
node --version
```

**2. 建立虛擬環境並安裝 Python 相依套件**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --pre -r requirements.txt
```

必須加上 `--pre`，因為針對 YouTube 的修正通常先以 yt-dlp nightly 版本發布。

## 工作流程

### 步驟 1 — 下載

```bash
python download.py [OPTIONS] URL [URL ...]
```

將 YouTube 播放清單中每部影片的音訊（m4a）、字幕（VTT）與中繼資料（info.json）下載到 `corpus/<video_id>/`。

| 選項 | 預設值 | 說明 |
|---|---|---|
| `-o, --output` | `corpus/` | 輸出目錄 |
| `--lang` | `zh-TW` | 字幕語言代碼 |
| `--audio-format` | `m4a` | 音訊格式（`m4a` 保留原始音訊串流；`wav`/`flac`/`mp3` 會重新編碼為 16 kHz 單聲道） |
| `--max-videos N` | 全部 | 下載 N 部影片後停止（測試用） |
| `--no-subs` | 關 | 不下載字幕 |

下載完成後會寫出 `corpus/manifest.csv`，列出每部影片的音訊路徑、字幕路徑，以及沒有字幕的影片會標記 `needs_whisper`。

**範例：**

```bash
# 下載播放清單的前 50 部影片
python download.py --max-videos 50 "https://www.youtube.com/playlist?list=..."

# 全部下載
python download.py "https://www.youtube.com/playlist?list=..."
```

**只要一部或幾部影片？** `fetch_youtube.py` 使用 `yt-dlp` CLI（`brew install yt-dlp`）產生相同的目錄結構，不需要 `.venv`。有上傳者字幕時優先使用，否則退回自動字幕（加 `--no-auto-subs` 可拒絕自動字幕）。若影片最後缺少音訊或字幕，會以非零狀態碼結束。以 `-` 開頭的影片 ID 也能正常處理。

```bash
python3 fetch_youtube.py "https://www.youtube.com/watch?v=_4SUBYKiN9E"
python3 fetch_youtube.py --info-json --lang 'zh-TW,zh-Hant' <url> [<url> ...]
```

下載後的語料目錄結構：

```
corpus/
├── manifest.csv
└── <video_id>/
    ├── <video_id>.m4a          # 音訊（原始 AAC 串流）
    ├── <video_id>.zh-TW.vtt   # 字幕
    └── <video_id>.info.json   # 中繼資料（標題、說明、頻道、上傳日期…）
```

---

### 步驟 2 — 打包

```bash
python pack.py [OPTIONS]
```

讀取語料目錄，以 ffmpeg 從 m4a 切出每個字幕段落，並寫成 parquet 檔。Schema 與 [ky552/ML2021_ASR_ST](https://huggingface.co/datasets/ky552/ML2021_ASR_ST) 相同。

YouTube CC 字幕每則約 1.7 秒，沒有標點，斷點落在畫面文字捲動之處——對 ASR 訓練而言太短，也常在句子中間斷開。切片前，`pack.py` 會執行 **LLM 輔助的重新分段**，將原始字幕組成完整語句，每組輸出一列 parquet。若設定了 `GEMINI_API_KEY`，由 Gemini（`gemini-3.1-flash-lite`）依語意句界分組；否則使用規則式的備案（`merge_segments()`），累積字幕至約 10 秒。

| 選項 | 預設值 | 說明 |
|---|---|---|
| `--corpus` | `corpus/` | 語料目錄 |
| `--output` | `dataset.parquet` | 輸出 parquet 檔 |
| `--max-videos N` | 全部 | 只打包 N 部影片（測試用） |

**範例：**

```bash
# 用 5 部影片測試
python pack.py --max-videos 5 --output test.parquet

# 完整打包
python pack.py --output dataset.parquet
```

Parquet schema：

| 欄位 | 型別 | 說明 |
|---|---|---|
| `video_id` | string | YouTube 影片 ID |
| `file` | string | 段落檔名（例如 `abc123_0001.wav`） |
| `audio` | struct `{bytes, path}` | 以位元組內嵌的片段：**原始 AAC 串流直接複製進 MP4**（`-c:a copy -f mp4`），未經解碼。通常為 44.1/48 kHz 立體聲；訓練前請重新取樣為 16 kHz 單聲道 |
| `transcription` | string | 此段落的字幕文字 |
| `start` | float32 | 段落起始時間（秒） |
| `end` | float32 | 段落結束時間（秒） |
| `title` | string | 影片標題 |
| `channel` | string | 頻道名稱 |
| `upload_date` | string | 上傳日期（`YYYYMMDD`） |

#### 不載入音訊快速檢視 parquet

`audio.bytes` 佔了檔案大部分的體積，所以只想看文字樣本時，要把音訊欄位排除才會快。`peek.sh` 就是做這件事——針對單一檔案或 glob，印出前 N 列的 `video_id  start-end  transcription`：

```bash
./peek.sh dataset.parquet               # 前 20 列
./peek.sh dataset.parquet 50            # 前 50 列
./peek.sh 'dataset_part_*.parquet' 100  # 跨所有分段檔的前 100 列
```

---

### 步驟 3 — 微調

將 `dataset.parquet` 傳到要進行微調的機器：

```bash
# 複製到遠端訓練伺服器
scp dataset.parquet user@train-server:/data/

# 或推送到 HuggingFace Hub
huggingface-cli upload <your-org>/<dataset-name> dataset.parquet
```

在 Python 中用 HuggingFace `datasets` 函式庫載入：

```python
from datasets import load_dataset

ds = load_dataset("parquet", data_files="dataset.parquet", split="train")
# ds[0]["audio"]          → {"array": np.ndarray, "sampling_rate": 16000, "path": "..."}
# ds[0]["transcription"]  → "今年以來的行情"
```

`datasets` 函式庫會自動將 `audio` 欄位解碼為 16 kHz 的 numpy 陣列，可搭配 Whisper、wav2vec2 及多數 ASR/TTS 微調框架使用。

---

## 之後新增影片（增量更新）

Gemini 分段與 parquet 打包都是完全增量的——新增 100 部影片大約只需 **0.10 美元的 Gemini 費用 + 約 20 分鐘的 ffmpeg**，不必整個重新打包。靠的是兩層快取：

- **每部影片的分段快取** — `corpus/<id>/<id>.zh-TW.segments.json` 每部影片只寫一次；之後執行 `pack.py` 會直接讀取，完全跳過 Gemini。
- **分段 parquet 續跑** — 設定 `--snapshot-every N` 時，`pack.py` 啟動會呼叫 `scan_completed_parts()`，讀取每個既有 `dataset_part_*.parquet` 的 `video_id`，並將這些 ID 從工作佇列移除。既有分段檔不會被動到；新資料寫入新的 `dataset_part_NNN.parquet`，從下一個可用編號開始。

### 流程

```bash
# 1. 下載新的影片 ID
.venv/bin/python download.py --cookies cookies.txt \
  --sleep-interval 4 --max-sleep-interval 10 \
  <NEW_VIDEO_URLS>

# 2. 重新執行 pack——會自動偵測哪些已完成
nohup bash -c 'set -a && source gemini-key.env && set +a && \
  .venv/bin/python -u pack.py --parallel 32 --workers 4 --snapshot-every 50 \
  --output dataset.parquet' > pack.log 2>&1 &

# pack.log 會顯示：
#   Resume: NNNN videos already packed in existing parts; continuing with part_NNN
#   Packing K videos → dataset.parquet           ← 只有新的影片
#   Pre-segment: K uncached videos, 32 concurrent Gemini calls.
```

### 重試下載失敗的影片

每次長時間執行結束後總會有一些失敗的 ID，而它們需要完全相反的處理方式。速率限制、機器人驗證、逾時和 503 之後會自己恢復；私人或已移除的影片則永遠不會。`classify_failures.py` 讀取 yt-dlp 日誌（由舊到新；每個 ID 以最後一筆錯誤為準），並寫出兩份清單。`--corpus` 會排除之後已成功下載的 ID，因為成功不會留下任何日誌行來覆蓋先前的錯誤：

```bash
python3 classify_failures.py download.log resume.log --corpus corpus \
  --retry pending_retry.txt --skip permanent_skip.txt
.venv/bin/python download.py --cookies cookies.txt --sleep-interval 4 --max-sleep-interval 10 \
  $(sed 's|^|https://www.youtube.com/watch?v=|' pending_retry.txt)
```

在本語料的三輪下載中，它產生的結果與手工整理的清單一致。約 90 個失敗 ID 中，33 個是永久失敗，其餘皆可重試。

### 注意事項

- **若重新下載了 VTT**（例如 YouTube 修正了字幕，或重新執行了 `align.py`），分段快取就過期了。刪除對應的 `<id>.zh-TW.segments.json` 以強制重新分段。已打包進 `dataset_part_*.parquet` 的資料也仍是依據舊 VTT——若要重建，請刪除含有該 `video_id` 的分段檔。
- **續跑是以 `video_id` 為鍵，而非分段策略。** 若更換策略（例如從 Gemini 切換為規則式，或改用 #2 的配對合併方式），必須刪除既有的分段檔。分段快取仍可省下 Gemini 費用，因此即使完全更換策略，API 成本也很低。

---

## 下載經驗談

整理自下載這份約 1,700 部影片的 zh-TW 語料（首次執行加兩輪續跑）。分段、字幕去重與時間戳偏移的問題，請參考*打包*一節與 `align.py`。

### 音訊

- **保留 m4a，下載時不要轉檔。** `-f bestaudio[ext=m4a]` 會逐位元組保存
  YouTube 的 AAC 串流。之後再依各使用端需求轉成 WAV：每套 ASR 工具要的格式
  都略有不同（取樣率、聲道數、RIFF 細節）。
- **m4a 通常是 44.1/48 kHz 立體聲**，不是 16 kHz 單聲道。我們因此遇到兩個問題：
  - Azure 批次轉錄處理立體聲檔時**會對兩個聲道各輸出一次語句**，
    所以每句都出現兩次。請先混成單聲道或只保留一個聲道（`-ac 1`）。
  - `pack.py` 是將 AAC 串流直接複製進每個 parquet 片段，所以 parquet 中的音訊
    **也不是** 16 kHz 單聲道 WAV（本 README 以前寫錯了）。下游使用者必須自行
    解碼與重新取樣。
- **絕對不要用 pipe 輸出 ffmpeg 的 WAV**（`-f wav pipe:1`）。pipe 無法 seek，
  ffmpeg 就沒辦法回頭填入 RIFF 與 `data` chunk 的大小，兩者都會停在
  `0xFFFFFFFF`。ffprobe 與 HuggingFace 不在意；但 Azure Custom Speech 會拒收
  每一個片段（"Duration in the wave file's header does not match the file size"）。
  一律寫到實際的檔案路徑。

### 字幕

- **使用 YouTube 實際提供的語言代碼。** 台灣頻道的手動字幕是 `zh-TW`，
  不是 `zh-Hant`/`zh`。不確定時請執行 `yt-dlp --list-subs <url>`。
  若各頻道不同，`--lang` 可接受逗號分隔的清單。
- **手動字幕優於自動字幕。** 使用 `--write-subs --write-auto-subs` 時，若該語言
  有上傳者提供的字幕，yt-dlp 會優先採用，否則才退回 ASR 自動字幕。若機器字幕
  會汙染參考資料集，請加 `--no-auto-subs`；自動字幕只是另一個 ASR 系統的
  假設，不是標準答案。
- **預期約 13% 的影片完全沒有字幕。** 在我們的語料中，1,545 部影片有 1,336 部
  具備 `zh-TW` 字幕。`download.py` 會在 `manifest.csv` 中為這些影片標記
  `needs_whisper`。

### 被封鎖（最花時間的部分）

- **現在必須有 JS 執行環境。** YouTube 的 n-sig 挑戰是透過 `yt-dlp-ejs` 破解，
  而 yt-dlp 預設只啟用 **deno**。若只裝了 node，會出現成千上萬條
  `No supported JavaScript runtime` 警告並缺少可用格式。請傳入
  `--js-runtimes node`（`download.py` 與 `fetch_youtube.py` 都已內建），或安裝 deno。
- **保持 yt-dlp 為最新版。** 針對 YouTube 的修正通常先出現在 nightly 版本，
  所以請用 `pip install --pre` 安裝。
  原本能下載的影片開始失敗時，先升級再除錯。典型症狀：字幕與 info.json 正常
  下載，但音訊出現 **`HTTP Error 403: Forbidden`**。2026 年 5 月的 nightly 版本
  到了 2026 年 11 月就是這樣；升級後即解決。這也是 `requirements.txt` 設定
  最低版本、而非鎖定特定版本的原因。
- **長時間執行要限速。** 不限速地跑播放清單會遇到 *"The current session has been
  rate-limited by YouTube for up to an hour"*。使用 `--sleep-interval 4
  --max-sleep-interval 10` 可穩定跑完 1,500 部以上的影片。被限速的 ID **並沒有**
  失效：一小時後再重試即可。
- **"Sign in to confirm you're not a bot"** 需要 cookies：從已登入的瀏覽器匯出
  Netscape 格式的 `cookies.txt`，並傳入 `--cookies`。**Cookies 會輪替。**
  某次續跑先出現 112 條 *"cookies are no longer valid"* 警告，接著是 53 次機器人
  驗證失敗。每次長時間執行前請重新匯出 cookies，並使用拋棄式帳號。
  **絕對不要提交 `cookies.txt`**（已列在 `.gitignore`）。
- **區分可重試與永久性失敗。** 三輪下來約 90 個失敗 ID 中：33 個是私人/已移除
  （永久失敗，永遠跳過），其餘是速率限制、機器人驗證、逾時或 503，多數在之後的
  輪次成功。`classify_failures.py` 會根據日誌自動分類。注意 *"Video
  unavailable. This content isn't available, try again later"* 其實是**速率限制**，
  雖然它以 "Video unavailable" 開頭。
- **忽略 `no impersonate target is available`。** 這個警告只代表沒有安裝
  `curl_cffi`，不影響下載。
- **完整記錄日誌**（`nohup … > download.log 2>&1 &`）。日誌是 ID *為何*失敗的
  唯一紀錄；成功不會留下任何行，所以之後的成功不會覆蓋先前的錯誤
  （`classify_failures.py --corpus` 會處理這點）。

### Shell 陷阱

- YouTube ID 可能以 `-` 開頭（例如 `-AbCdEfGhIj`）。在單純的 ID 前加上 `--`
  （`fetch_youtube.py` 內部已處理），並在 shell 迴圈中加上引號。
- URL 一律加引號：`&list=…` 會被 shell 當成背景執行運算子。
