import io
import json
import os
import tempfile
import time
import hmac
import wave
import audioop
import hashlib
import random
import base64
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st
from openai import OpenAI

try:
    from supabase import create_client
except Exception:
    create_client = None


# ============================================================
# Basic settings
# ============================================================
st.set_page_config(
    page_title="言いカエル おたすけAI",
    page_icon="🐸",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .block-container {
        max-width: 720px;
        padding-top: 1rem;
        padding-bottom: 4rem;
      }
      div.stButton > button {
        min-height: 3.25rem;
        border-radius: 16px;
        font-size: 1.05rem;
        font-weight: 650;
      }
      .game-card {
        border: 1px solid rgba(128,128,128,.24);
        border-radius: 20px;
        padding: 1rem 1.05rem;
        margin: .35rem 0 .8rem;
      }
      .topic-text {
        font-size: 1.35rem;
        line-height: 1.55;
        font-weight: 750;
      }
      .support-card {
        border: 1px solid rgba(128,128,128,.20);
        border-radius: 18px;
        padding: .9rem 1rem;
        margin: .4rem 0 .8rem;
      }
      .answer-card {
        border: 1px solid rgba(128,128,128,.20);
        border-radius: 18px;
        padding: .85rem 1rem;
        margin: .45rem 0;
      }
      .answer-main {
        font-size: 1.18rem;
        line-height: 1.55;
        font-weight: 750;
      }
      .small-note {
        font-size: .93rem;
        opacity: .82;
      }
      .answer-label-row {
        display: flex;
        align-items: center;
        gap: .5rem;
        margin-bottom: .15rem;
      }
      .answer-label-mini {
        font-size: .82rem;
        line-height: 1.2;
        font-weight: 700;
        opacity: .82;
      }
      .frog-badge {
        width: 1.65rem;
        height: 1.65rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        font-size: 1rem;
        border: 1px solid rgba(0,0,0,.08);
      }
      .frog-yellow { background: #FFF2A8; }
      .frog-blue { background: #CFE8FF; }
      .frog-pink { background: #FFD5E5; }
      .judge-card {
        border: 2px solid rgba(80,160,100,.35);
        border-radius: 20px;
        padding: 1rem 1.05rem;
        margin: .6rem 0 .8rem;
      }
      .judge-winner {
        font-size: 1.25rem;
        line-height: 1.5;
        font-weight: 800;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def secret(name, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


OPENAI_API_KEY = secret("OPENAI_API_KEY")
TEXT_MODEL = secret("TEXT_MODEL", "gpt-5.6-luna")
TRANSCRIBE_MODEL = secret("TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
TTS_MODEL = secret("TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = secret("TTS_VOICE", "coral")
IMAGE_MODEL = secret("IMAGE_MODEL", "gpt-image-2")
FAMILY_PIN = str(secret("FAMILY_PIN", "")).strip()
APP_TIMEZONE = secret("APP_TIMEZONE", "Asia/Tokyo")
SUPABASE_URL = secret("SUPABASE_URL", "")
SUPABASE_SECRET_KEY = secret("SUPABASE_SECRET_KEY", "")
USE_FAST_MODE = str(secret("USE_FAST_MODE", "true")).lower() in {"1", "true", "yes", "on"}


# ============================================================
# Card master (from the family-made list)
# ============================================================
TOPIC_CARDS = ['ゲーマー', '老後', 'ボランティア', '料理教室', '釣り', '旅行', 'ゲーム', '読書', '音楽', 'スポーツ観戦', 'パーティー', '会議', '面接', 'テスト', '夏休み', 'お正月', 'クリスマス', '誕生日', '葬式', '結婚式', 'デート', '通学', '買い物', '掃除', '料理', '植物', 'ペット', '隣人', '部下', '上司', 'ライバル', '後輩', '先輩', '親友', '家族', '努力家', '天才', 'ベジタリアン', '夜型の人', '早起きの人', 'コレクター', '受験生', 'アルバイト', '社長', 'ユーチューバー', 'アーティスト', 'プログラマー', '宇宙飛行士', '警察官', '医者', '教師', '学生', '主婦', '芸能人', 'スマートフォン', 'お金持ち', '自動車', '恋人', '痩せている人', '太っている人', 'ANY（出題者がお題を考えます）']
STYLE_CARDS = ['皮肉たっぷりに', '回りくどく', '映画・ドラマのセリフ風に', '歌の歌詞風に', '漫画・アニメ・ゲームに例えて', '動物に例えて', '可愛く', '子どもっぽく', '超前向きな言葉で', '四字熟語で', 'ストレートに', '怖い感じに', 'ギャル風に', '詩的に', '古風に', 'ファンタジー風に', '悲しい雰囲気で', 'カッコよく', '辛辣に', 'やさしく']
ANY_TOPIC_CARD = "ANY（出題者がお題を考えます）"


# ============================================================
# Clients / setup
# ============================================================
@st.cache_resource(show_spinner=False)
def openai_client():
    return OpenAI(api_key=OPENAI_API_KEY)


@st.cache_resource(show_spinner=False)
def supabase_client():
    if not (create_client and SUPABASE_URL and SUPABASE_SECRET_KEY):
        return None
    return create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)


def history_enabled():
    return bool(create_client and SUPABASE_URL and SUPABASE_SECRET_KEY)


def verify_setup():
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY が設定されていません。Streamlit の Secrets を確認してください。")
        st.stop()


def require_family_pin():
    if not FAMILY_PIN:
        return
    if st.session_state.get("_family_authenticated", False):
        return

    st.title("🐸 言いカエル おたすけAI")
    st.caption("家族用のあいことばを入れてください。")

    failures = int(st.session_state.get("_family_pin_failures", 0))
    locked_until = float(st.session_state.get("_family_pin_locked_until", 0.0))
    now = time.time()

    if locked_until > now:
        st.warning(f"入力回数が多いため、あと{max(1, int(locked_until - now))}秒ほど待ってください。")
        st.stop()

    entered = st.text_input(
        "あいことば",
        type="password",
        max_chars=32,
        key="_family_pin_input",
        autocomplete="off",
    )

    if st.button("はいる", type="primary", use_container_width=True):
        if entered and hmac.compare_digest(entered.strip(), FAMILY_PIN):
            st.session_state["_family_authenticated"] = True
            st.session_state["_family_pin_failures"] = 0
            st.session_state["_family_pin_locked_until"] = 0.0
            st.rerun()

        failures += 1
        if failures >= 5:
            st.session_state["_family_pin_failures"] = 0
            st.session_state["_family_pin_locked_until"] = time.time() + 60
            st.error("入力回数が多いため、1分ほど待ってからもう一度試してください。")
        else:
            st.session_state["_family_pin_failures"] = failures
            st.error("あいことばが違います。")
    st.stop()


# ============================================================
# OpenAI helpers
# ============================================================
def ask_json(prompt, name, schema, max_output_tokens=700):
    args = {
        "model": TEXT_MODEL,
        "input": prompt,
        "reasoning": {"effort": "none"},
        "text": {
            "format": {
                "type": "json_schema",
                "name": name,
                "strict": True,
                "schema": schema,
            }
        },
        "max_output_tokens": max_output_tokens,
        "store": False,
    }
    if USE_FAST_MODE:
        args["service_tier"] = "fast"
    result = openai_client().responses.create(**args)
    return json.loads(result.output_text)


def transcribe_audio(audio_file, context=""):
    audio_file.seek(0)
    prompt = (
        "5〜6歳の子どもの日本語の発話です。"
        "子どもらしい言い回しを勝手に大人の表現へ直しすぎず、"
        "聞こえた内容を自然な日本語として文字起こししてください。"
    )
    if context:
        prompt += " 文脈: " + context[:900]

    result = openai_client().audio.transcriptions.create(
        model=TRANSCRIBE_MODEL,
        file=audio_file,
        language="ja",
        prompt=prompt,
    )
    return result.text.strip()


def allowed_cards(card_type):
    return TOPIC_CARDS if card_type == "topic" else STYLE_CARDS


def transcribe_card_audio(audio_file, card_type):
    """Transcribe one card while strongly biasing recognition to the actual card master."""
    audio_file.seek(0)
    card_label = "お題カード" if card_type == "topic" else "言い方カード"
    master = allowed_cards(card_type)
    master_text = "、".join(master)
    prompt = (
        f"カードゲーム『言いカエル』の{card_label}を1枚、声で読んでいます。"
        "原則として次のカード一覧のどれかです。意味が近い別表現へ言い換えず、"
        "聞こえた発音そのものに最も近いカード語を意識して文字起こししてください。"
        "濁音・半濁音・長音・促音・拗音・助詞をできるだけ保ってください。"
        f" カード一覧: {master_text}"
    )
    result = openai_client().audio.transcriptions.create(
        model=TRANSCRIBE_MODEL,
        file=audio_file,
        language="ja",
        prompt=prompt,
    )
    return result.text.strip()


def boost_recorded_wav(audio_file):
    """Make quiet browser WAV recordings easier to transcribe."""
    audio_file.seek(0)
    original = audio_file.read()

    try:
        with wave.open(io.BytesIO(original), "rb") as reader:
            nchannels = reader.getnchannels()
            sampwidth = reader.getsampwidth()
            framerate = reader.getframerate()
            nframes = reader.getnframes()
            comptype = reader.getcomptype()
            compname = reader.getcompname()
            frames = reader.readframes(nframes)

        if not frames or sampwidth not in (1, 2, 3, 4):
            raise ValueError("Unsupported WAV format")

        rms = audioop.rms(frames, sampwidth)
        peak = audioop.max(frames, sampwidth)
        target_rms = 6000
        max_gain = 12.0
        peak_limit = 30000

        gain = 1.0 if rms <= 0 else max(1.0, min(max_gain, target_rms / rms))
        if peak > 0:
            gain = min(gain, max(1.0, peak_limit / peak))

        boosted = audioop.mul(frames, sampwidth, gain)
        out = io.BytesIO()
        with wave.open(out, "wb") as writer:
            writer.setnchannels(nchannels)
            writer.setsampwidth(sampwidth)
            writer.setframerate(framerate)
            writer.setnframes(nframes)
            writer.setcomptype(comptype, compname)
            writer.writeframes(boosted)
        out.seek(0)
        out.name = "recording_boosted.wav"
        return out
    except Exception:
        fallback = io.BytesIO(original)
        fallback.name = "recording.wav"
        return fallback


def speech_bytes(text):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = tmp.name

        with openai_client().audio.speech.with_streaming_response.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=text,
            instructions=(
                "5〜6歳の日本語話者の子どもに話しかけます。"
                "少しゆっくり、明瞭に、親しみやすく話してください。"
                "大げさな演技や過剰な感情表現は避けてください。"
                "言葉遊びの面白さは残し、短い文の間に自然な間を取ってください。"
            ),
            response_format="wav",
        ) as response:
            response.stream_to_file(temp_path)

        with open(temp_path, "rb") as f:
            return f.read()
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def generate_player_image(topic, style, label, answer, why):
    prompt = f"""
Create one square illustration for a Japanese family word-play card game.
The viewer is a 5- to 6-year-old child.

Topic card: {topic}
Style card: {style}
AI answer type: {label}
AI phrase: {answer}
Why it works: {why}

Requirements:
- Make the AI phrase immediately understandable as a funny visual scene.
- Strongly reflect the mood/form of the style card: {style}.
- Keep the humor playful, surprising, and easy for a young child to understand.
- Use a cute, friendly, colorful picture-book illustration style.
- If the style is scary, sarcastic, harsh, or sad, keep it child-safe and gentle rather than disturbing or cruel.
- Do not mock body shape, appearance, disability, race, gender, or other personal traits.
- Do not use copyrighted characters, brand mascots, logos, or recognizable franchises.
- Do not put words, captions, speech bubbles, letters, numbers, or logos in the image.
- One clear main scene, simple composition, easy to recognize on a phone screen.
""".strip()

    result = openai_client().images.generate(
        model=IMAGE_MODEL,
        prompt=prompt,
        size="1024x1024",
        quality="low",
    )
    if not result.data:
        raise ValueError("画像データが返りませんでした。")
    encoded = getattr(result.data[0], "b64_json", None)
    if not encoded:
        raise ValueError("画像データを読み取れませんでした。")
    return base64.b64decode(encoded)


def card_candidates(raw_text, card_type):
    master = allowed_cards(card_type)
    schema = {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "items": {"type": "string", "enum": master},
                "minItems": 1,
                "maxItems": 5,
            }
        },
        "required": ["candidates"],
        "additionalProperties": False,
    }

    master_lines = "\n".join(f"- {item}" for item in master)
    result = ask_json(
        f"""
カードゲーム『言いカエル』の音声認識結果を、実在するカード一覧と照合します。

【聞こえた文字】
{raw_text}

【実在するカード】
{master_lines}

【最重要ルール】
- 候補は上の実在カードからだけ選ぶ。新しい言葉を作らない。
- 意味が似ているかではなく、発音が似ているかだけで順位をつける。
- ひらがなで読んだときの音を比較する。
- 濁音/半濁音、長音、促音「っ」、拗音「ゃゅょ」、母音、1音程度の脱落・挿入、助詞の聞き違いを考慮する。
- 意味が近くても音が遠いカードは候補にしない。
- 最も音が近いカードを先頭にし、最大5件まで。
""".strip(),
        f"card_master_match_{card_type}",
        schema,
        max_output_tokens=220,
    )

    candidates = []
    for item in result.get("candidates", []):
        item = str(item).strip()
        if item in master and item not in candidates:
            candidates.append(item)

    raw = " ".join(str(raw_text or "").split()).strip()
    if raw in master:
        candidates = [raw] + [x for x in candidates if x != raw]

    return candidates[:5]


# ============================================================
# Game AI
# ============================================================
def support_answer(topic, style, child_request, level, previous_message=""):
    word_note_schema = {
        "type": "object",
        "properties": {
            "word": {"type": "string"},
            "meaning": {"type": "string"},
            "image": {"type": "string"},
        },
        "required": ["word", "meaning", "image"],
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {
            "need_type": {
                "type": "string",
                "enum": ["word_meaning", "wording", "idea", "mixed"],
            },
            "message": {"type": "string"},
            "word_notes": {
                "type": "array",
                "items": word_note_schema,
                "minItems": 0,
                "maxItems": 2,
            },
        },
        "required": ["need_type", "message", "word_notes"],
        "additionalProperties": False,
    }

    level_rules = {
        1: """
第1段階。原則として完成した回答は言わない。
- 単語の意味を聞かれた場合は、その単語の意味を短く説明してよい。
- 言い回しや発想で困っている場合は、見方を1つ変えるヒントだけ出す。
- 動物、乗り物、食べ物、ヒーロー、身近な物など、頭に絵が浮かぶ例を使う。
""",
        2: """
第2段階。完成した回答はまだ言わない。
- 使えそうな言葉を1〜2語だけ渡す。
- 新しい語は、意味と頭に浮かぶ具体的な場面を短く添える。
- 子ども自身が組み合わせられる余地を残す。
""",
        3: """
第3段階。答えの骨組みまで助ける。
- 「○○＋勇者」「○○みたいな△△」など、組み立て方を示してよい。
- 完成形を複数並べない。穴を少し残して子どもが最後を決められるようにする。
""",
        4: """
第4段階。かなり困っているので、短い完成例を1つだけ出してよい。
- その例をそのまま使うことを求めず、「ここから変えてもいい」と分かる言い方にする。
- できれば、別の表現を作るための材料となる言葉を1語だけ添える。
""",
    }

    return ask_json(
        f"""
あなたは、5〜6歳の子どもがカードゲーム「言いカエル」で困ったときだけ使うサポート役です。
辞書のように説明を並べるのではなく、子どもが頭に場面を思い浮かべて、自分で言い換えを作れるように助けます。

【今回のお題】
{topic}

【指定された言い方】
{style}

【子どもが話したこと】
{child_request}

【今回のヒント段階】
{level}/4

【前回のヒント】
{previous_message or '（なし）'}

【この段階のルール】
{level_rules.get(level, level_rules[4])}

【共通ルール】
- AIはゲームの主役にならない。目的は子ども自身が回答を思いつくこと。
- 返答は短く、音声で聞いて分かる自然な日本語にする。
- 「正解」「間違い」「すごい」「えらい」などの評価は不要。
- 必要以上に褒めたり感情的に盛り上げたりしない。
- 子どもに親しみやすく、少しおかしみのある具体例は使ってよい。
- 難しい熟語を連発しない。新しい語彙は最大2語。
- 人を傷つける、容姿をばかにする、差別的な言い方は避ける。
- お題に人の弱点が含まれていても、その特徴を別の見方に変えて遊ぶ。
- AIプレイヤーの回答は知らない前提で、このサポートだけを独立して行う。
- message は基本1〜3文程度。
- word_notes の meaning は子ども向けに短く。
- word_notes の image は「大きなゾウがゆっくり歩く感じ」のように、頭に絵が浮かぶ一言にする。
""".strip(),
        f"support_level_{level}",
        schema,
        max_output_tokens=650,
    )


def player_style_instruction(style):
    rules = {
        "皮肉たっぷりに": "表面ではほめているように聞こえるが、少しだけ反対の意味がにじむ言い方にする。子ども同士で傷つける悪口にはしない。",
        "回りくどく": "結論をすぐ言わず、前置きや遠回りを入れてから意味が伝わる言い方にする。短すぎてストレートにならないようにする。",
        "映画・ドラマのセリフ風に": "登場人物が場面の中で実際に口にしそうなセリフにする。『〜だ』『〜なのか』『行こう』など、声に出して演じられる形を優先する。",
        "歌の歌詞風に": "リズムや繰り返し、情景のある言葉を使い、歌に乗せられそうな一節にする。説明文にはしない。",
        "漫画・アニメ・ゲームに例えて": "漫画・アニメ・ゲームの世界にありそうな役、技、アイテム、イベントなどにたとえる。作品固有名を無理に使わず、子どもが絵を想像できる言い方にする。",
        "動物に例えて": "必ず動物そのもの、または動物の動き・特徴を使ってたとえる。動物が出てこない回答は禁止。",
        "可愛く": "語感、擬音、小ささ、丸さ、やわらかさなどを使い、聞いた瞬間に『かわいい』と感じる言い方にする。",
        "子どもっぽく": "5〜6歳の子どもが実際に言いそうな、短く素朴で具体的な言葉にする。大人っぽい熟語や抽象表現は避ける。",
        "超前向きな言葉で": "必ず明るい長所・チャンス・楽しみとして言い換える。否定的な語感を残さない。",
        "四字熟語で": "回答そのものを原則として漢字4文字の四字熟語にする。既存の四字熟語を優先し、難しすぎる場合は子どもにも説明できるものを選ぶ。",
        "ストレートに": "遠回しにせず、一言で意味がはっきり伝わる直接的な表現にする。比喩を使う場合も意味がすぐ分かるものにする。",
        "怖い感じに": "怪物、闇、危機、ぞくっとする音などを使い、少し怖い雰囲気がはっきり出る言い方にする。ただし残酷な描写はしない。",
        "ギャル風に": "明るくテンポよく、現代のくだけた若者口調にする。『マジ』『めっちゃ』『〜じゃん』などは使ってよいが、難しいネットスラングにはしない。",
        "詩的に": "情景、光、風、季節、色、音などを使い、説明ではなく少し余韻のある表現にする。",
        "古風に": "現代の普通の言い方を避け、『〜でござる』『〜なり』『いざ』『〜じゃ』など昔風の響きを明確に入れる。",
        "ファンタジー風に": "魔法、勇者、王国、竜、精霊、宝物、冒険など、幻想世界を連想できる要素を必ず入れる。",
        "悲しい雰囲気で": "寂しさ、別れ、涙、静けさなどが感じられる言い方にする。暗すぎたり不安を強く煽ったりしない。",
        "カッコよく": "強さ、速さ、頼もしさ、特別感が伝わる言い方にする。聞いた瞬間にヒーローや達人のような印象が出ることを優先する。",
        "辛辣に": "少し鋭く、遠慮のないツッコミ調にする。ただし人そのものを傷つける悪口や容姿いじりにはしない。",
        "やさしく": "やわらかい語調で、安心・思いやり・温かさが伝わる言い方にする。強い否定や命令口調を避ける。",
    }
    return rules.get(style, "言い方カードの語感・形式・雰囲気が、回答だけを聞いても明確に伝わるようにする。")


def player_answers(topic, style):
    style_instruction = player_style_instruction(style)
    answer_schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "why": {"type": "string"},
            "new_word": {"type": "string"},
            "new_word_meaning": {"type": "string"},
        },
        "required": ["answer", "why", "new_word", "new_word_meaning"],
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {
            "metaphor": answer_schema,
            "nickname": answer_schema,
            "twist": answer_schema,
        },
        "required": ["metaphor", "nickname", "twist"],
        "additionalProperties": False,
    }

    return ask_json(
        f"""
あなたは、5〜6歳の子どもと父母が遊ぶカードゲーム「言いカエル」の4人目のプレイヤーです。
今回のお題を、指定された言い方に言い換えます。

【お題】
{topic}

【言い方カード】
{style}

【この言い方カードで必ず守ること】
{style_instruction}

【最重要】
- 最終回答 answer は「ワンフレーズ」または「短いツーフレーズ」にする。説明文や長い文章は禁止。
- 原則は1フレーズ。2フレーズにするのは、前振り→オチ、反復、対比などで面白さが明確に増す場合だけ。
- answer 全体は短く、声に出して一息〜二息で言える長さにする。目安は合計30文字以内。
- 「お題に合っていること」よりも、「言い方カードにしっかり寄っていること」を強く優先する。
- 3つのカエルは発想の方向を変えるための補助軸であり、言い方カードより優先してはいけない。
- 回答だけを聞いた人が、言い方カードを知らなくても「これは『{style}』っぽい」と当てられるくらい、語尾・語彙・リズム・雰囲気・形式をはっきり寄せる。
- 言い方カードの特徴は why で説明するのではなく、必ず answer そのものに出す。
- 「無難で上手」より「短くて、ちょっと意外で、子どもが絵を思い浮かべて笑える」を優先する。
- 子ども向けの笑いは、①予想外の組み合わせ、②少し大げさ、③音やリズムの面白さ、④頭に浮かぶ変な光景、⑤意外な逆転、のうち少なくとも1つを使う。
- ただし意味不明なランダム語、悪口、容姿いじり、下品すぎる表現にはしない。
- 「すごい○○」「○○名人」「○○ヒーロー」のような無難な名付けだけで終わらせない。使うなら、具体的で意外な修飾を足して一段ひねる。
- 出力前に3回答それぞれを内部で確認し、①短いか、②『{style}』らしさが一発で分かるか、③5〜6歳が絵を想像できるか、④少し笑える意外性があるか、のどれかが弱ければ作り直してから出力する。

必ず次の3方向で、互いに違う回答を1つずつ作ってください。

1. metaphor = 「たとえカエル」
   動物、乗り物、食べ物、ヒーロー、怪獣、自然、身近な物などにたとえる。
   頭に絵が浮かぶことを最優先する。

2. nickname = 「なまえカエル」
   新しいあだ名、称号、キャラクター名のような言い方にする。
   声に出したとき楽しく、子どもがまねしやすいものにする。

3. twist = 「ぎゃくてんカエル」
   一見弱点や困った特徴でも、別の角度から見て長所・面白さ・役割へ変える。
   説教ではなく、意外だけれど意味が通る表現にする。

【回答品質の優先順位】
1. 言い方カード「{style}」らしさが、answerだけで明確に出ている。
2. 1〜2フレーズで短く、声に出した瞬間に通じる。
3. 5〜6歳が頭の中に変な・楽しい場面を描ける。
4. 「そう来たか」と感じる小さな意外性があり、子どもが笑いやすい。
5. お題とのつながりが自然で、意味が破綻していない。

【短くても言い方カードを強く出すための型】
- 皮肉たっぷりに：ほめる形なのに少し逆の意味がにじむ短い一言。
- 回りくどく：短い前置き＋短い結論の2フレーズまで。
- 映画・ドラマのセリフ風に：その場で人物が叫ぶ・つぶやく短いセリフ。
- 歌の歌詞風に：リズム、反復、擬音などで短い歌詞の一節。
- 漫画・アニメ・ゲームに例えて：「必殺！」「レベル○○」「伝説のアイテム」など世界観が一発で出る型。
- 動物に例えて：answer内に具体的な動物名を必ず出す。
- 可愛く：擬音、語尾、小ささ、丸さなどをanswer内に出す。
- 子どもっぽく：短く素朴に。「○○だー！」「めっちゃ○○！」のような勢いも可。
- 超前向きな言葉で：「チャンス」「ラッキー」「最高」など明るい転換をanswer内に出す。
- 四字熟語で：原則、answerは漢字4文字そのもの。
- ストレートに：一発で意味が分かる短い断言。
- 怖い感じに：「闇」「怪物」「ぞくっ」「来る…」など怖さがanswer内に出る。
- ギャル風に：「マジ」「めっちゃ」「〜じゃん」など口調をanswer内に出す。
- 詩的に：光、風、月、雨、色、音などの具体的な情景を短く置く。
- 古風に：「いざ」「〜なり」「〜でござる」「〜じゃ」などをanswer内に出す。
- ファンタジー風に：魔法、勇者、竜、精霊、王国などをanswer内に出す。
- 悲しい雰囲気で：涙、さよなら、ひとり、しょんぼり等の寂しさをanswer内に出す。
- カッコよく：必殺技・異名・強い動きのような切れ味を出す。
- 辛辣に：短いツッコミとして鋭く。ただし人格攻撃にはしない。
- やさしく：安心できる短い言葉、柔らかな語尾をanswer内に出す。

【ルール】
- 3回答は発想を重複させない。3つとも同じ「○○名人」「○○ヒーロー」型にしない。
- answer は説明を含めない。説明は why に分離する。
- answer は原則1フレーズ、最大2フレーズ。1〜2文の長文にはしない。
- 「回りくどく」でも短い2フレーズ内で回りくどさを表現する。
- 「映画・ドラマのセリフ風に」「歌の歌詞風に」「詩的に」でも一息〜二息の短さを守る。
- 難解な熟語、抽象語、難しいネットスラング、大人しか分からない皮肉は避ける。
- ダジャレだけに頼らないが、音の面白さが自然なら使ってよい。
- 人を傷つける、容姿をばかにする、差別的な表現は避ける。
- お題に人の弱点が含まれる場合も、その人を笑うのではなく、特徴の見方や状況をずらして笑えるようにする。
- why は子ども向けに1文で「どこが言い方カードに合っていて、どこが面白いか」が分かるようにする。
- new_word は、その回答に少し新しい語彙が含まれる場合だけ1語。不要なら空文字。
- new_word_meaning は new_word が空なら空文字。ある場合は子ども向けに非常に短く説明する。
""".strip(),
        "player_answers",
        schema,
        max_output_tokens=700,
    )


def player_speech_text(answers):
    return (
        f"たとえカエル。{answers['metaphor']['answer']}。"
        f"その理由は、{answers['metaphor']['why']}。"
        f"なまえカエル。{answers['nickname']['answer']}。"
        f"その理由は、{answers['nickname']['why']}。"
        f"ぎゃくてんカエル。{answers['twist']['answer']}。"
        f"その理由は、{answers['twist']['why']}。"
    )


def is_meaningful_judge_answer(answer):
    """Exclude only obvious non-answers; keep ordinary weak/short answers eligible."""
    raw = str(answer or "").strip()
    compact = "".join(ch for ch in raw if ch not in " 　、。,.!?！？…・〜ー~")
    if len(compact) < 2:
        return False

    lowered = compact.lower()
    non_answers = [
        "わからない", "分からない", "わかんない", "わかりません",
        "思いつかない", "おもいつかない", "思いつきません",
        "ありません", "ないです", "なし", "無回答", "パス",
        "むり", "無理", "知らない", "しらない",
    ]
    return not any(phrase in lowered for phrase in non_answers)


def is_child_player_name(name):
    """Return True for ordinary labels used for a child player in family play."""
    normalized = str(name or "").strip().lower().replace(" ", "").replace("　", "")
    child_markers = [
        "こども", "子ども", "子供", "お子さん", "おこさん",
        "息子", "むすこ", "娘", "むすめ", "キッズ", "kid", "child",
    ]
    return any(marker.lower() in normalized for marker in child_markers)


def judge_answers(topic, style, players):
    eligible = [p for p in players if is_meaningful_judge_answer(p.get("answer", ""))]
    if not eligible:
        raise ValueError("判定できる回答がありません。『わからない』『パス』以外の回答を1つ以上入れてください。")

    # Family-play weighting:
    # - If a child has a meaningful answer, the child side wins about 50% of rounds.
    # - The remaining ~50% is chosen randomly from the other meaningful answers.
    # - With multiple children, the 50% child share is split randomly among them.
    # - Obvious non-answers never receive the child weighting.
    rng = random.SystemRandom()
    child_eligible = [p for p in eligible if is_child_player_name(p.get("name", ""))]
    other_eligible = [p for p in eligible if p not in child_eligible]

    if child_eligible and other_eligible:
        if rng.random() < 0.5:
            winner = rng.choice(child_eligible)
        else:
            winner = rng.choice(other_eligible)
    else:
        winner = rng.choice(eligible)

    schema = {
        "type": "object",
        "properties": {
            "reason": {"type": "string"},
        },
        "required": ["reason"],
        "additionalProperties": False,
    }
    result = ask_json(
        f"""
あなたは、家族で遊ぶカードゲーム『言いカエル』の審判です。
今回はゲームとして1位の人はすでに抽選で決まっています。
あなたの役目は、その回答の良いところを見つけて、子どもにも分かる理由を1文で説明することだけです。
順位を選び直したり、他の回答と比較したりしないでください。

【お題カード】
{topic}

【言い方カード】
{style}

【今回1位になった人】
{winner['name']}

【今回1位になった回答】
{winner['answer']}

【理由の作り方】
- 指定された『言い方』とのつながり、発想の面白さ、イメージしやすさなどから、実際に当てはまる良い点を1つ見つける。
- 無理に大げさに褒めない。
- 他の人の回答には触れない。
- 難しい言葉は使わない。
- 1文だけにする。
- 『〜からです。』で終える。
""".strip(),
        "judge_reason",
        schema,
        max_output_tokens=180,
    )

    return {
        "winner_id": winner["id"],
        "winner_name": winner["name"],
        "winner_answer": winner["answer"],
        "reason": result["reason"],
        "eligible_count": len(eligible),
    }


def judge_speech_text(result):
    return (
        f"今回いちばん良かったのは、{result['winner_name']}の、"
        f"『{result['winner_answer']}』です。"
        f"理由は、{result['reason']}"
    )


# ============================================================
# Optional Supabase history
# ============================================================
def today():
    return datetime.now(ZoneInfo(APP_TIMEZONE)).date().isoformat()


def create_round_history(topic, style, ai_joined):
    if not history_enabled():
        return None
    try:
        result = (
            supabase_client()
            .table("iikaeru_rounds")
            .insert(
                {
                    "played_date": today(),
                    "topic": topic,
                    "style": style,
                    "ai_joined": bool(ai_joined),
                }
            )
            .execute()
        )
        return result.data[0]["id"] if result.data else None
    except Exception:
        return None


def update_round_history(round_id, **fields):
    if not (round_id and history_enabled() and fields):
        return
    try:
        (
            supabase_client()
            .table("iikaeru_rounds")
            .update(fields)
            .eq("id", round_id)
            .execute()
        )
        recent_rounds.clear()
    except Exception:
        pass


@st.cache_data(ttl=30, show_spinner=False)
def recent_rounds(limit=30):
    if not history_enabled():
        return []
    result = (
        supabase_client()
        .table("iikaeru_rounds")
        .select(
            "id,played_date,topic,style,ai_joined,ai_answers,ai_revealed,"
            "support_log,learned_words"
        )
        .order("played_date", desc=True)
        .order("id", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data


# ============================================================
# Session state
# ============================================================
DEFAULTS = {
    "round_active": False,
    "round_id": None,
    "topic": "",
    "style": "",
    "ai_joined": False,
    "ai_answers": None,
    "ai_revealed": False,
    "ai_audio": None,
    "ai_audio_autoplay_pending": False,
    "ai_images": {},
    "support_open": False,
    "support_request": "",
    "support_level": 0,
    "support_result": None,
    "support_audio": None,
    "support_autoplay_pending": False,
    "support_log": [],
    "learned_words": [],
    "judge_result": None,
    "judge_audio_bytes": None,
    "judge_autoplay_pending": False,
    "judge_signature": "",
    "round_serial": 0,
}


def fresh_default(value):
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    return value


for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = fresh_default(value)


def reset_round():
    serial = int(st.session_state.get("round_serial", 0)) + 1
    for key, value in DEFAULTS.items():
        st.session_state[key] = fresh_default(value)
    st.session_state.round_serial = serial
    for key in list(st.session_state.keys()):
        if key.startswith((
            "take_", "transcript_", "review_audio_",
            "topic_draft_", "style_draft_", "topic_audio_", "style_audio_",
            "_topic_", "_style_", "_topic_draft_", "_style_draft_",
            "score_player_count_", "score_name_", "score_audio_",
            "score_answer_", "score_digest_",
        )):
            del st.session_state[key]


def voice_review(key, title, context=""):
    take_key = "take_" + key
    if take_key not in st.session_state:
        st.session_state[take_key] = 0
    take = st.session_state[take_key]

    transcript_key = f"transcript_{key}_{take}"
    audio_key = f"review_audio_{key}_{take}"

    st.markdown("### " + title)
    audio = st.audio_input(
        "マイクを押して話してね",
        sample_rate=16000,
        key=f"audio_{key}_{take}",
    )

    if audio is not None and transcript_key not in st.session_state:
        try:
            boosted = boost_recorded_wav(audio)
            audio_bytes = boosted.getvalue()
            st.session_state[audio_key] = audio_bytes

            audio_for_transcription = io.BytesIO(audio_bytes)
            audio_for_transcription.name = "recording_boosted.wav"
            with st.spinner("声を文字にしています…"):
                transcript = transcribe_audio(audio_for_transcription, context)
            if transcript:
                st.session_state[transcript_key] = transcript
            else:
                st.warning("うまく聞き取れなかったよ。もう一度話してね。")
        except Exception as exc:
            st.error("声の聞き取りに失敗しました。もう一度試してください。")
            with st.expander("保護者向け詳細"):
                st.code(str(exc))

    transcript = str(st.session_state.get(transcript_key, "")).strip()
    if not transcript:
        return None

    st.markdown("**こう聞こえたよ**")
    st.info(transcript)

    c1, c2 = st.columns(2)
    with c1:
        proceed = st.button(
            "この内容で すすむ",
            type="primary",
            use_container_width=True,
            key=f"proceed_{key}_{take}",
        )
    with c2:
        retry = st.button(
            "もういちど 話す",
            use_container_width=True,
            key=f"retry_{key}_{take}",
        )

    if retry:
        st.session_state.pop(transcript_key, None)
        st.session_state.pop(audio_key, None)
        st.session_state[take_key] += 1
        st.rerun()

    if proceed:
        return transcript
    return None


def audio_digest(uploaded_file):
    if uploaded_file is None:
        return ""
    try:
        uploaded_file.seek(0)
        data = uploaded_file.read()
        uploaded_file.seek(0)
        return hashlib.sha1(data).hexdigest()
    except Exception:
        return ""


def voice_select_card(field_key, audio_key, label, card_type, context=""):
    transcript_key = f"_{field_key}_transcript"
    candidates_key = f"_{field_key}_candidates"
    digest_key = f"_{audio_key}_digest"
    radio_key = f"_{field_key}_radio"
    manual_key = f"_{field_key}_manual"

    st.markdown(f"**{label}**")
    audio = st.audio_input(
        f"🎤 {label}を読んでね",
        sample_rate=16000,
        key=audio_key,
    )

    if audio is not None:
        digest = audio_digest(audio)
        if digest and st.session_state.get(digest_key) != digest:
            try:
                boosted = boost_recorded_wav(audio)
                audio_for_transcription = io.BytesIO(boosted.getvalue())
                audio_for_transcription.name = "recording_boosted.wav"
                with st.spinner(f"{label}を聞いています…"):
                    transcript = transcribe_card_audio(audio_for_transcription, card_type)
                    candidates = card_candidates(transcript, card_type) if transcript else []

                if transcript and candidates:
                    st.session_state[transcript_key] = transcript
                    st.session_state[candidates_key] = candidates
                    st.session_state[digest_key] = digest
                    st.session_state.pop(radio_key, None)
                    st.session_state.pop(manual_key, None)
                    st.rerun()
                else:
                    st.warning(f"{label}をうまく聞き取れませんでした。もう一度話してください。")
            except Exception as exc:
                st.error(f"{label}の聞き取りに失敗しました。もう一度試してください。")
                with st.expander("保護者向け詳細"):
                    st.code(str(exc))

    candidates = list(st.session_state.get(candidates_key, []))
    if not candidates:
        st.caption("カードを声で読むと、聞き取り候補がここに出ます。")
        return ""

    transcript = str(st.session_state.get(transcript_key, "")).strip()
    if transcript:
        st.caption(f"いちばん近く聞こえた音：{transcript}")

    options = candidates + ["どれもちがう"]
    selected = st.radio(
        "実際のカードに書いてある言葉を選んでね",
        options,
        key=radio_key,
    )

    if selected == "どれもちがう":
        manual = st.text_input(
            "カードに書いてある言葉を入力",
            key=manual_key,
            placeholder="候補にないときだけ入力",
        )
        return str(manual or "").strip()

    return str(selected or "").strip()


def voice_capture_player_answer(round_serial, index, player_name, topic, style):
    answer_key = f"score_answer_{round_serial}_{index}"
    digest_key = f"score_digest_{round_serial}_{index}"
    audio_key = f"score_audio_{round_serial}_{index}"

    st.markdown(f"**{player_name} の回答**")
    audio = st.audio_input(
        f"🎤 {player_name} が答える",
        sample_rate=16000,
        key=audio_key,
    )

    if audio is not None:
        digest = audio_digest(audio)
        if digest and st.session_state.get(digest_key) != digest:
            try:
                boosted = boost_recorded_wav(audio)
                audio_for_transcription = io.BytesIO(boosted.getvalue())
                audio_for_transcription.name = "answer.wav"
                context = (
                    f"カードゲーム『言いカエル』です。お題は『{topic}』、"
                    f"言い方は『{style}』。{player_name}が考えた短い回答を話しています。"
                    "回答の言葉を勝手に別表現へ直さず、聞こえた通りに文字起こししてください。"
                )
                with st.spinner(f"{player_name} の答えを聞いています…"):
                    transcript = transcribe_audio(audio_for_transcription, context)
                if transcript:
                    st.session_state[answer_key] = transcript
                    st.session_state[digest_key] = digest
            except Exception as exc:
                st.error(f"{player_name} の回答を聞き取れませんでした。")
                with st.expander("保護者向け詳細"):
                    st.code(str(exc))

    if answer_key not in st.session_state:
        st.session_state[answer_key] = ""

    answer = st.text_input(
        f"{player_name} の聞き取り結果",
        key=answer_key,
        placeholder="音声を入れるとここに表示されます",
    )
    return str(answer or "").strip()


def support_speech_text(result):
    parts = [str(result.get("message", "")).strip()]
    notes = result.get("word_notes", []) or []
    if notes:
        parts.append("ことばメモも聞いてね。")
        for note in notes:
            word = str(note.get("word", "")).strip()
            meaning = str(note.get("meaning", "")).strip()
            image = str(note.get("image", "")).strip()
            if word:
                parts.append(f"{word}。")
            if meaning:
                parts.append(f"意味は、{meaning}。")
            if image:
                parts.append(f"たとえば、{image}。")
    return " ".join(part for part in parts if part)


def apply_support_result(child_request, level, result):
    st.session_state.support_request = child_request
    st.session_state.support_level = level
    st.session_state.support_result = result
    st.session_state.support_audio = speech_bytes(support_speech_text(result))
    st.session_state.support_autoplay_pending = True

    event = {
        "request": child_request,
        "level": level,
        "need_type": result.get("need_type", ""),
        "message": result.get("message", ""),
        "words": result.get("word_notes", []),
    }
    st.session_state.support_log.append(event)

    learned = list(st.session_state.learned_words)
    for note in result.get("word_notes", []):
        word = str(note.get("word", "")).strip()
        if word and word not in learned:
            learned.append(word)
    st.session_state.learned_words = learned

    update_round_history(
        st.session_state.round_id,
        support_log=st.session_state.support_log,
        learned_words=learned,
    )


def render_support_result():
    result = st.session_state.support_result
    if not result:
        return

    st.markdown(
        f"""
        <div class="support-card">
          <b>ヒント {st.session_state.support_level}/4</b><br><br>
          {result['message']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.support_audio:
        st.audio(
            st.session_state.support_audio,
            format="audio/wav",
            autoplay=bool(st.session_state.support_autoplay_pending),
        )
        st.session_state.support_autoplay_pending = False

    notes = result.get("word_notes", [])
    if notes:
        st.markdown("**ことばメモ**")
        for note in notes:
            st.markdown(
                f"- **{note['word']}**：{note['meaning']}  \\n  *{note['image']}*"
            )


def render_player_answers(answers, allow_image_generation=False):
    labels = [
        ("metaphor", "たとえカエル", "frog-yellow"),
        ("nickname", "なまえカエル", "frog-blue"),
        ("twist", "ぎゃくてんカエル", "frog-pink"),
    ]
    for key, label, color_class in labels:
        item = answers[key]
        st.markdown(
            f"""
            <div class="answer-card">
              <div class="answer-label-row">
                <span class="frog-badge {color_class}">🐸</span>
                <span class="answer-label-mini">{label}</span>
              </div>
              <div class="answer-main">{item['answer']}</div>
              <div class="small-note">{item['why']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if item.get("new_word"):
            st.caption(f"ことばメモ：{item['new_word']} ＝ {item['new_word_meaning']}")

        if allow_image_generation:
            images = st.session_state.get("ai_images", {})
            existing = images.get(key)
            button_label = "🎨 この答えを絵にする" if not existing else "🎨 もう一度絵にする"
            if st.button(
                button_label,
                use_container_width=True,
                key=f"ai_image_{st.session_state.round_serial}_{key}",
            ):
                try:
                    with st.spinner("絵を作っています…"):
                        image_bytes = generate_player_image(
                            st.session_state.topic,
                            st.session_state.style,
                            label,
                            item["answer"],
                            item["why"],
                        )
                    updated = dict(st.session_state.get("ai_images", {}))
                    updated[key] = image_bytes
                    st.session_state.ai_images = updated
                    st.rerun()
                except Exception as exc:
                    st.error("絵を作れませんでした。もう一度試してください。")
                    with st.expander("保護者向け詳細"):
                        st.code(str(exc))

            existing = st.session_state.get("ai_images", {}).get(key)
            if existing:
                st.image(
                    existing,
                    caption=f"{label}：{item['answer']}",
                    use_container_width=True,
                )


# ============================================================
# Main UI
# ============================================================
verify_setup()
require_family_pin()

st.title("🐸 言いカエル おたすけAI")
st.caption("困ったときはヒント係。AIもプレイヤー。最後はAIが審判もできます。")
st.caption("※ 読み上げ音声はAIが生成した音声です。")

pages = ["あそぶ", "これまで"] if history_enabled() else ["あそぶ"]
page = st.radio("画面", pages, horizontal=True, label_visibility="collapsed")

if page == "これまで":
    st.subheader("これまでの言いカエル")
    try:
        rows = recent_rounds(30)
        if not rows:
            st.info("まだ記録はありません。")
        for row in rows:
            title = f"{row.get('played_date', '')}　{row.get('topic', '')} → {row.get('style', '')}"
            with st.expander(title):
                learned = row.get("learned_words") or []
                support_log = row.get("support_log") or []
                st.write(f"サポートを使った回数：{len(support_log)}")
                if learned:
                    st.write("出会ったことば：" + "・".join(learned))
                if row.get("ai_revealed") and row.get("ai_answers"):
                    st.markdown("**AIの3回答**")
                    render_player_answers(row["ai_answers"])
    except Exception as exc:
        st.error("履歴を読み込めませんでした。Supabase の設定を確認してください。")
        with st.expander("保護者向け詳細"):
            st.code(str(exc))
    st.stop()


if not st.session_state.round_active:
    st.subheader("カードをセット")
    st.caption("登録済みのカード一覧から、声の音に近い実在カードだけを候補表示します。実物と同じ言葉を選んでください。")

    base_context = "カードゲーム『言いカエル』のカードを読み上げています。短い語句として、聞こえた音をできるだけそのまま文字起こししてください。"
    topic = voice_select_card(
        field_key=f"topic_draft_{st.session_state.round_serial}",
        audio_key=f"topic_audio_{st.session_state.round_serial}",
        label="お題カード",
        card_type="topic",
        context=base_context + " 今はお題カードです。",
    )
    style = voice_select_card(
        field_key=f"style_draft_{st.session_state.round_serial}",
        audio_key=f"style_audio_{st.session_state.round_serial}",
        label="言い方カード",
        card_type="style",
        context=base_context + " 今は言い方カードです。",
    )
    if topic == ANY_TOPIC_CARD:
        topic = st.text_input(
            "ANYのお題",
            placeholder="出題者が考えた今回のお題",
            key=f"any_topic_{st.session_state.round_serial}",
        ).strip()
    ai_join = st.checkbox("AIもこのラウンドに参加する", value=True)

    if st.button("このお題ではじめる", type="primary", use_container_width=True):
        topic = str(topic).strip()
        style = str(style).strip()
        if not topic or not style:
            st.warning("お題カードと言い方カードを両方入れてください。")
        else:
            try:
                round_id = create_round_history(topic, style, ai_join)
                st.session_state.round_id = round_id
                st.session_state.topic = topic
                st.session_state.style = style
                st.session_state.ai_joined = bool(ai_join)
                st.session_state.ai_answers = None
                st.session_state.ai_revealed = False

                st.session_state.round_active = True
                st.rerun()
            except Exception as exc:
                st.error("ゲームを始められませんでした。設定を確認してください。")
                with st.expander("保護者向け詳細"):
                    st.code(str(exc))
    st.stop()


# Active round
st.markdown(
    f"""
    <div class="game-card">
      <div class="small-note">お題</div>
      <div class="topic-text">{st.session_state.topic}</div>
      <br>
      <div class="small-note">こんな言い方に</div>
      <div class="topic-text">{st.session_state.style}</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------- Support mode --------------------
st.subheader("① こまったときのサポート")
st.caption("分からない言葉や、思いつかないところを声で聞けます。AIはまず答えを言わずに助けます。")

if not st.session_state.support_open:
    if st.button("🐸 こまった！ たすけて", use_container_width=True):
        st.session_state.support_open = True
        st.rerun()
else:
    if not st.session_state.support_request:
        context = (
            f"カードゲームの今回のお題は『{st.session_state.topic}』、"
            f"言い方は『{st.session_state.style}』です。"
        )
        child_request = voice_review(
            f"support_{st.session_state.round_serial}",
            "なにで こまっているか話してね",
            context,
        )
        if child_request is not None:
            try:
                with st.spinner("ヒントを考えています…"):
                    result = support_answer(
                        st.session_state.topic,
                        st.session_state.style,
                        child_request,
                        level=1,
                    )
                    apply_support_result(child_request, 1, result)
                st.rerun()
            except Exception as exc:
                st.error("ヒントを作れませんでした。もう一度試してください。")
                with st.expander("保護者向け詳細"):
                    st.code(str(exc))
    else:
        st.caption("聞いたこと：" + st.session_state.support_request)
        render_support_result()

        c1, c2 = st.columns(2)
        with c1:
            more_disabled = st.session_state.support_level >= 4
            if st.button(
                "もう少しヒント",
                use_container_width=True,
                disabled=more_disabled,
            ):
                try:
                    new_level = min(4, st.session_state.support_level + 1)
                    previous = st.session_state.support_result.get("message", "")
                    with st.spinner("もう一歩だけ助けます…"):
                        result = support_answer(
                            st.session_state.topic,
                            st.session_state.style,
                            st.session_state.support_request,
                            level=new_level,
                            previous_message=previous,
                        )
                        apply_support_result(
                            st.session_state.support_request,
                            new_level,
                            result,
                        )
                    st.rerun()
                except Exception as exc:
                    st.error("ヒントを作れませんでした。もう一度試してください。")
                    with st.expander("保護者向け詳細"):
                        st.code(str(exc))
        with c2:
            if st.button("別のことを聞く", use_container_width=True):
                st.session_state.support_request = ""
                st.session_state.support_level = 0
                st.session_state.support_result = None
                st.session_state.support_audio = None
                st.session_state.support_autoplay_pending = False
                # Force a new recorder key.
                old_key = f"take_support_{st.session_state.round_serial}"
                st.session_state[old_key] = int(st.session_state.get(old_key, 0)) + 1
                st.rerun()

    if st.button("サポートを閉じる", use_container_width=True):
        st.session_state.support_open = False
        st.rerun()


st.divider()


# -------------------- Judging mode --------------------
st.subheader("② AI審判・採点モード")
st.caption("明らかな無回答を除いて判定します。子どもが有効な回答をしている回は、子どもが約50%の確率で1位になります。残りはほかの参加者からランダムに選び、AIが良かった理由を説明します。点数は付けません。")

serial = st.session_state.round_serial
player_count = st.selectbox(
    "人間の参加人数",
    [2, 3, 4, 5],
    index=1,
    key=f"score_player_count_{serial}",
)

default_names = ["こども", "お父さん", "お母さん", "プレイヤー4", "プレイヤー5"]
players = []
for i in range(int(player_count)):
    name_key = f"score_name_{serial}_{i}"
    if name_key not in st.session_state:
        st.session_state[name_key] = default_names[i]
    name = st.text_input(
        f"{i + 1}人目の名前",
        key=name_key,
    ).strip() or f"プレイヤー{i + 1}"
    answer = voice_capture_player_answer(
        serial, i, name, st.session_state.topic, st.session_state.style
    )
    players.append({"id": f"P{i + 1}", "name": name, "answer": answer})

ready = all(p["answer"] for p in players)
current_signature = hashlib.sha1(
    json.dumps(
        {
            "topic": st.session_state.topic,
            "style": st.session_state.style,
            "players": players,
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()

if st.session_state.judge_result and st.session_state.judge_signature != current_signature:
    st.session_state.judge_result = None
    st.session_state.judge_audio_bytes = None
    st.session_state.judge_autoplay_pending = False
    st.session_state.judge_signature = ""

if not ready:
    st.caption("全員の回答が入ると判定できます。")

if st.button(
    "🏆 AIに今回の1位を決めてもらう",
    type="primary",
    use_container_width=True,
    disabled=not ready,
):
    try:
        with st.spinner("みんなの答えを比べています…"):
            result = judge_answers(
                st.session_state.topic,
                st.session_state.style,
                players,
            )
            audio_bytes = speech_bytes(judge_speech_text(result))
        st.session_state.judge_result = result
        st.session_state.judge_audio_bytes = audio_bytes
        st.session_state.judge_autoplay_pending = True
        st.session_state.judge_signature = current_signature
        st.rerun()
    except ValueError as exc:
        st.warning(str(exc))
    except Exception as exc:
        st.error("判定できませんでした。もう一度試してください。")
        with st.expander("保護者向け詳細"):
            st.code(str(exc))

if st.session_state.judge_result:
    result = st.session_state.judge_result
    st.markdown(
        f"""
        <div class="judge-card">
          <div class="small-note">今回の1位</div>
          <div class="judge-winner">🏆 {result['winner_name']}</div>
          <div class="answer-main">「{result['winner_answer']}」</div>
          <br>
          <div><b>理由：</b>{result['reason']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.session_state.judge_audio_bytes:
        st.audio(
            st.session_state.judge_audio_bytes,
            format="audio/wav",
            autoplay=bool(st.session_state.judge_autoplay_pending),
        )
        st.session_state.judge_autoplay_pending = False


st.divider()

# -------------------- AI player mode --------------------
st.subheader("③ AIもゲームに参加")
st.caption("AIは『たとえ・なまえ・ぎゃくてん』の3方向で答えます。")

if not st.session_state.ai_joined:
    if st.button("AIもこのラウンドに参加", use_container_width=True):
        try:
            with st.spinner("AIもこっそり考えています…"):
                answers = player_answers(st.session_state.topic, st.session_state.style)
            st.session_state.ai_joined = True
            st.session_state.ai_answers = answers
            update_round_history(
                st.session_state.round_id,
                ai_joined=True,
                ai_answers=answers,
            )
            st.rerun()
        except Exception as exc:
            st.error("AIの回答を作れませんでした。もう一度試してください。")
            with st.expander("保護者向け詳細"):
                st.code(str(exc))
else:
    if not st.session_state.ai_revealed:
        if st.session_state.ai_answers is None:
            st.info("AIはまだ答えを作っていません。見るときに3つまとめて考えるので、はじめの読み込みが速くなっています。")
        else:
            st.info("AIも3つ考えました。まだ答えは伏せています。")

        if st.button("AIの答えを見る", type="primary", use_container_width=True):
            try:
                if st.session_state.ai_answers is None:
                    with st.spinner("AIが3つ考えています…"):
                        answers = player_answers(st.session_state.topic, st.session_state.style)
                    st.session_state.ai_answers = answers
                    update_round_history(
                        st.session_state.round_id,
                        ai_joined=True,
                        ai_answers=answers,
                    )
                st.session_state.ai_revealed = True
                update_round_history(st.session_state.round_id, ai_revealed=True)
                st.rerun()
            except Exception as exc:
                st.error("AIの回答を作れませんでした。もう一度試してください。")
                with st.expander("保護者向け詳細"):
                    st.code(str(exc))
    else:
        render_player_answers(st.session_state.ai_answers, allow_image_generation=True)
        if not st.session_state.ai_audio:
            if st.button("🔊 AIの答えを聞く", use_container_width=True):
                try:
                    with st.spinner("声を作っています…"):
                        st.session_state.ai_audio = speech_bytes(
                            player_speech_text(st.session_state.ai_answers)
                        )
                    st.session_state.ai_audio_autoplay_pending = True
                    st.rerun()
                except Exception as exc:
                    st.error("読み上げ音声を作れませんでした。")
                    with st.expander("保護者向け詳細"):
                        st.code(str(exc))
        else:
            st.audio(
                st.session_state.ai_audio,
                format="audio/wav",
                autoplay=bool(st.session_state.ai_audio_autoplay_pending),
            )
            st.session_state.ai_audio_autoplay_pending = False


st.divider()
if st.session_state.learned_words:
    st.caption("このラウンドで出会ったことば：" + "・".join(st.session_state.learned_words))

if st.button("つぎのお題へ", type="primary", use_container_width=True):
    reset_round()
    st.rerun()
