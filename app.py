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
# v19: one pedagogical AI reference answer + random card draw + four-section layout.
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
CARD_TABLE = secret("CARD_TABLE", "iikaeru_cards")
USE_FAST_MODE = str(secret("USE_FAST_MODE", "true")).lower() in {"1", "true", "yes", "on"}


# ============================================================
# Card master is stored privately in Supabase.
# No card text is embedded in this public GitHub source.
# ============================================================


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


@st.cache_data(ttl=300, show_spinner=False)
def load_card_rows():
    """Load the private card master with a server-side Supabase secret key."""
    client = supabase_client()
    if client is None:
        return []

    result = (
        client
        .table(CARD_TABLE)
        .select("card_type,card_text,is_free_topic,sort_order,ai_instruction")
        .eq("is_active", True)
        .order("card_type")
        .order("sort_order")
        .execute()
    )

    cleaned = []
    for row in result.data or []:
        card_type = str(row.get("card_type", "")).strip()
        card_text = str(row.get("card_text", "")).strip()
        if card_type not in {"topic", "style"} or not card_text:
            continue
        cleaned.append(
            {
                "card_type": card_type,
                "card_text": card_text,
                "is_free_topic": bool(row.get("is_free_topic", False)),
                "sort_order": int(row.get("sort_order") or 0),
                "ai_instruction": str(row.get("ai_instruction") or "").strip(),
            }
        )
    return cleaned


def allowed_cards(card_type):
    target = "topic" if card_type == "topic" else "style"
    return [
        row["card_text"]
        for row in load_card_rows()
        if row["card_type"] == target
    ]


def is_free_topic_card(card_text):
    target = str(card_text or "").strip()
    return any(
        row["card_type"] == "topic"
        and row["card_text"] == target
        and row["is_free_topic"]
        for row in load_card_rows()
    )


def history_enabled():
    return bool(create_client and SUPABASE_URL and SUPABASE_SECRET_KEY)


def verify_setup():
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY が設定されていません。Streamlit の Secrets を確認してください。")
        st.stop()

    if create_client is None:
        st.error("Supabase ライブラリがありません。requirements.txt に supabase を追加してください。")
        st.stop()

    if not (SUPABASE_URL and SUPABASE_SECRET_KEY):
        st.error(
            "カード一覧は非公開Supabaseから読み込みます。Streamlit Secrets に "
            "SUPABASE_URL と SUPABASE_SECRET_KEY を設定してください。"
        )
        st.stop()

    try:
        topics = allowed_cards("topic")
        styles = allowed_cards("style")
    except Exception as exc:
        st.error(
            "Supabase の非公開カード一覧を読み込めませんでした。"
            "テーブル設定とSecret Keyを確認してください。"
        )
        with st.expander("保護者向け詳細"):
            st.code(str(exc))
        st.stop()

    if not topics or not styles:
        st.error(
            "Supabase のカード一覧が空です。お題カードと言い方カードを登録してください。"
        )
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


def explain_topic_and_style_for_child(topic, style):
    """Explain the topic card and the style card in simple Japanese without giving a game answer."""
    schema = {
        "type": "object",
        "properties": {
            "topic_meaning": {"type": "string"},
            "style_meaning": {"type": "string"},
        },
        "required": ["topic_meaning", "style_meaning"],
        "additionalProperties": False,
    }

    style_instruction = player_style_instruction(style)
    base_prompt = f"""
あなたは、5〜6歳の子どもにカードゲーム「言いカエル」のカードの意味をやさしく説明する先生役です。

【今回のお題カード】
{topic}

【今回の言い方カード】
{style}

【言い方カードの内部方針】
{style_instruction}

【目的】
子どもが、まず「お題カードは何のことか」を頭に絵として思い浮かべ、次に「言い方カードは、どんな感じの言い方を求めているのか」を理解できるようにしてください。
ここではゲームの完成回答は作りません。お題の答え、言い換えの完成例、勝ち方につながる具体的な答えは出さず、カードの意味だけを説明します。

【ルール】
- 5〜6歳に話しかける自然な日本語にする。
- topic_meaning は1〜2文。難しい言葉を使わず、お題がどんなもの・人・場面かを説明する。
- style_meaning は1〜2文。言い方カードが「どんな見方・雰囲気・表し方を求めているか」を子ども向けに説明する。
- style_meaning では、今回のお題に対する完成回答や、そのまま使える言い換え例は絶対に出さない。
- 言い方カードの表面的な語尾・擬音だけを説明するのではなく、内部方針を踏まえて「どういう見方をする言い方なのか」をやさしく説明する。
- 全体を音声で30秒前後で聞ける程度に短くする。
- 子どもを評価したり褒めたりしない。
- 日本語の漢字・ひらがな・カタカナ・数字・一般的な句読点だけを使う。
- 外国語や日本語以外の文字体系、アルファベットを混ぜない。
""".strip()

    last_result = None
    for attempt in range(3):
        retry = ""
        if attempt:
            retry = (
                "\n\n前の出力に日本語以外の表記が含まれていた可能性があります。"
                "日本語表記だけで、より短く分かりやすく作り直してください。"
            )
        result = ask_json(
            base_prompt + retry,
            "topic_and_style_explanation",
            schema,
            max_output_tokens=480,
        )
        last_result = result
        joined = " ".join(
            str(result.get(key, ""))
            for key in ("topic_meaning", "style_meaning")
        )
        if not non_japanese_letters(joined):
            return result

    return last_result or {"topic_meaning": "", "style_meaning": ""}


def topic_explanation_speech_text(item):
    topic_meaning = str((item or {}).get("topic_meaning", "")).strip()
    style_meaning = str((item or {}).get("style_meaning", "")).strip()
    parts = []
    if topic_meaning:
        parts.append(f"まず、お題のせつめい。{topic_meaning}")
    if style_meaning:
        parts.append(f"つぎに、言い方のせつめい。{style_meaning}")
    return " ".join(parts)


def log_topic_explanation(item):
    """Keep the existing per-round support history useful after simplifying the UI."""
    message = topic_explanation_speech_text(item)
    event = {
        "request": "お題と言い方を解説して",
        "level": 0,
        "need_type": "topic_and_style_explanation",
        "message": message,
        "words": [],
    }
    st.session_state.support_log.append(event)
    update_round_history(
        st.session_state.round_id,
        support_log=st.session_state.support_log,
        learned_words=st.session_state.learned_words,
    )


def player_style_instruction(style):
    """Load style-specific AI guidance from the private Supabase card master."""
    target = str(style or "").strip()
    for row in load_card_rows():
        if row["card_type"] == "style" and row["card_text"] == target:
            instruction = str(row.get("ai_instruction") or "").strip()
            if instruction:
                return instruction
    return "言い方カードの語感・形式・雰囲気が、回答だけを聞いても明確に伝わるようにする。"


def style_logic_mode(style, style_instruction):
    """Choose a generation strategy from the selected style and its private guidance."""
    text = f"{style}\n{style_instruction}"
    # Sarcasm needs a two-layer meaning. Detect the concept rather than hard-coding
    # the full private card master in the public app.
    if "皮肉" in text or ("ほめ" in text and ("本当の意味" in text or "裏" in text)):
        return "sarcasm"
    return "general"


def sarcasm_generation_rules():
    return """
【皮肉たっぷり専用ロジック】
この言い方では「やさしいウィット」だけでは不合格です。
聞いた瞬間に「ほめてるようで、ちゃんと刺してる」と分かる、短く切れ味のある皮肉にしてください。

【皮肉の芯】
- 表：ほめる、感心する、ありがたがる、立派な肩書きを与える、など一見プラスの形を取る。
- 裏：実際には、お題の「やりすぎ」「困る結果」「矛盾」「空回り」「ズレ」をはっきり指す。
- オチ：できれば後半の短い言葉で意味をひっくり返す。説明しすぎず、最後に小さく刺す。
- 皮肉の対象は、人そのものではなく、その人の行動・状況・出来事を優先する。

【切れ味を出す4つの型】
1. ほめ殺し型：普通なら困る点を、わざと大げさな長所としてほめる。
2. 結果ツッコミ型：立派そうに言ったあと、その結果起きる困りごとを一言で置く。
3. 逆肩書き型：行動のズレが一瞬で見える、少し意地のある肩書きにする。
4. 無表情型：感情的に悪く言わず、事実を淡々と並べることで可笑しさを出す。

【作る順番】
1. お題の特徴を1つだけ選ぶ。
2. その特徴が行きすぎると何が困るのか、具体的な結果を1つ決める。
3. その困る結果を、わざと「ほめ言葉・感心・肩書き」の形で包む。
4. 前半より後半を強くし、最後の数語にオチを置く。
5. 内部では少なくとも12案を比較し、「皮肉の明確さ」「ウィット」「切れ味」が最も高い案を残す。

【優しすぎ判定】
次のどれかに当てはまる案は捨てる。
- そのまま本気のほめ言葉として受け取れてしまう。
- 「ちょっと面白い」だけで、困る点や矛盾が見えない。
- 最後にフォローして丸く収めてしまう。
- 「すごい」「さすが」「最高」を付けただけ。
- whyを読まないと皮肉だと分からない。

【禁止】
- 人格否定、容姿いじり、差別、侮辱、残酷な表現。
- 子どもがまねして相手を傷つけるような直接的な悪口。
- 長い説明文。

各候補について内部確認用に次も作ること：
- surface_meaning：表向きには何をほめているように聞こえるか。
- hidden_meaning：本当はどんな困りごと・矛盾・やりすぎを指しているか。

最終条件：answer単体で皮肉が分かり、短く、少し意地があり、でも悪口ではないこと。
""".strip()


def review_sarcasm_player_answers(topic, result):
    """Reject soft, merely funny, or non-ironic answers in a strict second pass."""
    review_item = {
        "type": "object",
        "properties": {
            "pass": {"type": "boolean"},
            "irony_score": {"type": "integer", "minimum": 0, "maximum": 3},
            "wit_score": {"type": "integer", "minimum": 0, "maximum": 3},
            "cut_score": {"type": "integer", "minimum": 0, "maximum": 3},
            "reason": {"type": "string"},
        },
        "required": ["pass", "irony_score", "wit_score", "cut_score", "reason"],
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {
            "metaphor": review_item,
            "nickname": review_item,
            "twist": review_item,
        },
        "required": ["metaphor", "nickname", "twist"],
        "additionalProperties": False,
    }
    compact = {
        key: {
            "answer": str((result.get(key, {}) or {}).get("answer", "")),
            "surface_meaning": str((result.get(key, {}) or {}).get("surface_meaning", "")),
            "hidden_meaning": str((result.get(key, {}) or {}).get("hidden_meaning", "")),
        }
        for key in ("metaphor", "nickname", "twist")
    }
    review = ask_json(
        f"""
あなたは子ども向け言葉ゲームの厳しい「皮肉たっぷり判定係」です。
お題は「{topic}」です。次の3回答を判定してください。

{json.dumps(compact, ensure_ascii=False)}

0〜3点で採点：
- irony_score：ほめる形と本当の意味が明確に逆向きか。
- wit_score：ただの悪口ではなく、「そう来たか」と思える知的なひねりがあるか。
- cut_score：優しすぎず、短い言葉の中にキレや小さな毒があるか。

合格条件：
- answer単体で皮肉が分かる。
- 表向きはプラスに見えるのに、裏では困る結果・矛盾・やりすぎを具体的に刺している。
- できれば後半にオチがあり、最後の数語で意味がひっくり返る。
- 本気のほめ言葉としてそのまま成立するなら不合格。
- 単なるジョーク、かわいい言い換え、比喩、大げさ表現だけなら不合格。
- 「やさしくまとめた」「最後にフォローした」ために毒が消えている場合は不合格。
- 人格・容姿・属性を攻撃する悪口は不合格。
- 行動や状況を切る皮肉は可。
- 5〜6歳でも説明を聞けば意味が分かる。

pass=true は irony_score、wit_score、cut_score がすべて2以上の場合だけにしてください。
reasonは短い日本語1文にしてください。
""".strip(),
        "sarcasm_review",
        schema,
        max_output_tokens=500,
    )
    problems = []
    for key in ("metaphor", "nickname", "twist"):
        item = review.get(key, {}) or {}
        scores_ok = all(int(item.get(score, 0) or 0) >= 2 for score in ("irony_score", "wit_score", "cut_score"))
        if not item.get("pass", False) or not scores_ok:
            reason = str(item.get("reason", "皮肉のキレが弱い")).strip()
            problems.append(f"{key}:{reason}")
    return problems


def review_sarcasm_reference(topic, result):
    schema = {
        "type": "object",
        "properties": {
            "pass": {"type": "boolean"},
            "irony_score": {"type": "integer", "minimum": 0, "maximum": 3},
            "wit_score": {"type": "integer", "minimum": 0, "maximum": 3},
            "cut_score": {"type": "integer", "minimum": 0, "maximum": 3},
            "reason": {"type": "string"},
        },
        "required": ["pass", "irony_score", "wit_score", "cut_score", "reason"],
        "additionalProperties": False,
    }
    review = ask_json(
        f"""
あなたは子ども向け言葉ゲームの厳しい「皮肉たっぷり判定係」です。
お題は「{topic}」です。
参考回答は「{result.get('answer', '')}」です。
表向きの意味は「{result.get('surface_meaning', '')}」。
裏の意味は「{result.get('hidden_meaning', '')}」。

次を0〜3点で判定してください。
- irony_score：表と裏がちゃんと逆向きか。
- wit_score：「そう来たか」と思えるひねりがあるか。
- cut_score：優しすぎず、短いキレや小さな毒があるか。

本気のほめ言葉として成立してしまう、ただ面白いだけ、最後にフォローして丸くなっている、answer単体では皮肉が分からない場合は不合格です。
一方で、人格・容姿・属性への悪口も不合格です。行動や状況の矛盾を短く切るものを評価してください。
pass=true は3項目すべて2点以上の場合だけです。
""".strip(),
        "sarcasm_reference_review",
        schema,
        max_output_tokens=260,
    )
    scores_ok = all(int(review.get(score, 0) or 0) >= 2 for score in ("irony_score", "wit_score", "cut_score"))
    return bool(review.get("pass", False) and scores_ok), str(review.get("reason", "")).strip()


def non_japanese_letters(text):
    """Return letter characters that are not Japanese kana/kanji.

    Punctuation, digits, spaces, and symbols are allowed. Alphabetic characters
    from Latin, Devanagari, Cyrillic, Hangul, Arabic, etc. are rejected so that
    player-facing AI text stays in Japanese notation.
    """
    bad = []
    for ch in str(text or ""):
        code = ord(ch)
        is_hiragana = 0x3040 <= code <= 0x309F
        is_katakana = 0x30A0 <= code <= 0x30FF
        is_halfwidth_katakana = 0xFF65 <= code <= 0xFF9F
        is_cjk = (
            0x3400 <= code <= 0x4DBF
            or 0x4E00 <= code <= 0x9FFF
            or 0xF900 <= code <= 0xFAFF
        )
        is_japanese_mark = 0x3000 <= code <= 0x303F

        if is_hiragana or is_katakana or is_halfwidth_katakana or is_cjk or is_japanese_mark:
            continue
        if ch.isspace() or ch.isdigit():
            continue
        # Punctuation / symbols (。！？「」・〜、 etc.) are fine.
        import unicodedata
        category = unicodedata.category(ch)
        if category.startswith("P") or category.startswith("S"):
            continue
        # Any remaining Unicode letter is a non-Japanese writing system.
        if category.startswith("L") or category.startswith("M"):
            bad.append(ch)
    return bad


def player_text_has_foreign_script(result):
    fields = []
    for key in ("metaphor", "nickname", "twist"):
        item = result.get(key, {}) or {}
        fields.extend([
            item.get("answer", ""),
            item.get("why", ""),
            item.get("new_word", ""),
            item.get("new_word_meaning", ""),
        ])
    return any(non_japanese_letters(value) for value in fields)


def player_answers(topic, style):
    style_instruction = player_style_instruction(style)
    logic_mode = style_logic_mode(style, style_instruction)
    answer_properties = {
        "answer": {"type": "string"},
        "why": {"type": "string"},
        "new_word": {"type": "string"},
        "new_word_meaning": {"type": "string"},
    }
    answer_required = ["answer", "why", "new_word", "new_word_meaning"]
    if logic_mode == "sarcasm":
        answer_properties.update({
            "surface_meaning": {"type": "string"},
            "hidden_meaning": {"type": "string"},
        })
        answer_required.extend(["surface_meaning", "hidden_meaning"])

    answer_schema = {
        "type": "object",
        "properties": answer_properties,
        "required": answer_required,
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

    base_prompt = f"""
あなたは、5〜6歳の子どもと父母が遊ぶカードゲーム「言いカエル」の4人目のプレイヤーです。
このゲームで大切なのは、単に言葉を派手に飾ることではありません。
お題の「見方」を少しずらし、聞いた人が「なるほど、そう来たか」と思って笑える、短く知的な言い換えを作ってください。

【お題】
{topic}

【言い方カード】
{style}

【この言い方カードの意味・狙い】
{style_instruction}

{sarcasm_generation_rules() if logic_mode == "sarcasm" else ""}

【回答を作る基本思想】
- 最優先は「一休さん的なウィット」。普通の見方をそのまま言わず、前提・役割・長所短所・場面・たとえのどれかを1回ひっくり返す。
- 面白さは、意味不明な奇抜さではなく「意外なのに、言われると筋が通っている」ことから作る。
- 5〜6歳でも、聞いた瞬間に具体的な場面や絵が浮かぶ言葉を使う。
- 言い方カードは、単なる語尾・擬音・飾りではなく「どう見せ直すか」の条件として使う。
- 擬音、決まり文句、派手な形容詞、語尾だけを付けてカードらしく見せるのは禁止。それだけでは言い換えになっていない。
- 「最強」「伝説」「ヒーロー」「王様」「達人」「マスター」などの便利な称号を足すだけの回答は禁止。使う場合も、その称号自体にお題との意外で筋の通った関係が必要。
- answer は原則ワンフレーズ、最大でも短いツーフレーズ。一息〜二息、合計30文字以内を目安にする。
- お題カードの文言「{topic}」を answer にそのまま使わない。別の言葉・比喩・役割・情景に変換する。
- 出力する answer・why・new_word・new_word_meaning は、すべて日本語表記だけにする。
- 英語・ヒンディー語・中国語の簡体字表現・韓国語など、外国語や日本語以外の文字体系を混ぜない。
- アルファベットも原則使わない。外来語は日本で一般的なカタカナ表記に直す。
- 日本語の漢字・ひらがな・カタカナ・数字・一般的な句読点だけで、5〜6歳が読んだり聞いたりできる形にする。

【内部での作り方】
最終回答を出す前に、内部では次の順で考える。途中案は出力しない。
1. お題について、子どもでも分かる「普通の見方・特徴」を2〜4個考える。
2. そのうち1つを、逆から見る・役割を変える・大げさに具体化する・別の物に置き換える・意外な長所として見る、のいずれかでずらす。
3. その発想を言い方カード「{style}」に合う形へ整える。
4. 少なくとも8案を頭の中で比較し、「カード適合」「ウィット」「分かりやすさ」「短さ」「子どもの笑いやすさ」が高い3案だけを残す。
5. ただの擬音、ただの派手語、ただの語尾変更、ただの称号になっている案は捨てる。

【3つの方向】
1. metaphor = 「たとえカエル」
   お題の特徴を、別の物・動物・乗り物・食べ物・自然・身近な場面などに置き換える。
   似ているだけではなく、「そのたとえ方があったか」と思える1段のひねりを入れる。

2. nickname = 「なまえカエル」
   あだ名・呼び名・肩書きのような形にする。
   ただ格好いい名前を付けるのではなく、その名前を聞くとお題の特徴が別の見方で見えてくるようにする。

3. twist = 「ぎゃくてんカエル」
   欠点→長所、困りごと→役割、普通→特別など、意味の向きを反転させる。
   「そう考えれば確かに」と納得できる逆転にする。

【言い方カードへの寄せ方】
- カードが「口調・形式」を求める場合は、その形式を守る。ただし形式だけで終わらず、内容にもひねりを入れる。
- カードが「雰囲気・印象」を求める場合は、擬音や形容詞を足すのではなく、お題の見え方そのものをその印象へ変える。
- カードが「何かに例える」ことを求める場合は、対象を出すだけではなく、お題との意外で分かりやすい共通点を作る。
- カードが「前向き・やさしい・鋭い」など評価の方向を求める場合は、評価語を直接足さず、そう感じる理由がフレーズの中に見えるようにする。

【最終チェック】
3回答それぞれについて、以下を満たさなければ内部で作り直す。
- 言い方カードを外しても成立するただの飾り言葉になっていないか。
- 「なぜこの言葉？」に1文で筋の通った説明ができるか。
- 5〜6歳が絵を思い浮かべられるか。
- 大人が聞いても「少しうまい」と感じるひねりがあるか。
- 子どもが笑える余地があるか。
- お題カードの文言「{topic}」をそのまま使っていないか。
- 日本語以外の文字体系や外国語表記が1文字でも混ざっていないか。混ざっていたら日本語に直してから出力する。
- 3つの発想が重複していないか。

【出力ルール】
- answer は説明文にしない。説明は why に分ける。
- why は子ども向けに1文で、「どこをどう見方を変えたから面白いか」が分かるようにする。
- 難解な熟語、抽象語、ネットスラングは避ける。皮肉モードでは、5〜6歳にも説明できる二重の意味を使う。
- ダジャレは、意味も通る場合だけ使う。音が似ているだけのダジャレは避ける。
- 人を傷つける、容姿をばかにする、差別的な表現は避ける。
- お題に人の弱点が含まれていても、その人を笑うのではなく「見方のずらし」で笑いを作る。
- new_word は、その回答に少し新しい語彙が含まれる場合だけ1語。不要なら空文字。
- new_word_meaning は new_word が空なら空文字。ある場合は子ども向けに非常に短く説明する。
""".strip()

    def topic_is_reused(answer):
        answer_text = str(answer or "").replace(" ", "").replace("　", "")
        topic_text = str(topic or "").replace(" ", "").replace("　", "")
        return bool(topic_text and topic_text in answer_text)

    last_result = None
    last_problems = []
    for attempt in range(4):
        retry_note = ""
        if attempt > 0:
            problems = "、".join(last_problems) if last_problems else "出力条件"
            retry_note = (
                "\n\n【作り直し指示】\n"
                f"前の回答は『{problems}』の条件に違反していました。"
                f"お題カードの文言『{topic}』をそのまま使わず、"
                "日本語以外の文字・外国語・アルファベットを一切混ぜず、"
                "外来語が必要ならカタカナ表記にし、"
                "表面的な擬音・派手語・語尾変更にも逃げず、"
                "見方を1回ずらした、短く筋の通るウィットへ作り直してください。"
            )
            if logic_mode == "sarcasm":
                retry_note += (
                    " さらに今回は皮肉なので、ただ面白いだけでは不可です。"
                    "表ではほめているように聞こえ、裏では困る点や矛盾を短く刺してください。最後にフォローして丸めず、answer単体で皮肉と分かるキレを残してください。"
                )

        result = ask_json(
            base_prompt + retry_note,
            "player_answers",
            schema,
            max_output_tokens=700,
        )
        last_result = result

        answers = [
            result.get("metaphor", {}).get("answer", ""),
            result.get("nickname", {}).get("answer", ""),
            result.get("twist", {}).get("answer", ""),
        ]
        last_problems = []
        if any(topic_is_reused(answer) for answer in answers):
            last_problems.append("お題の文言をそのまま使用")
        if player_text_has_foreign_script(result):
            last_problems.append("日本語以外の文字を使用")

        if logic_mode == "sarcasm":
            for key in ("metaphor", "nickname", "twist"):
                item = result.get(key, {}) or {}
                if not str(item.get("surface_meaning", "")).strip() or not str(item.get("hidden_meaning", "")).strip():
                    last_problems.append(f"{key}:表と裏の意味が不足")
            if not last_problems:
                last_problems.extend(review_sarcasm_player_answers(topic, result))

        if not last_problems:
            return result

    raise ValueError("日本語だけで、お題の文言をそのまま使わない回答を作れませんでした。もう一度AIの回答を生成してください。")


def reference_answer(topic, style):
    """Create one pedagogical reference answer with a child-friendly explanation."""
    style_instruction = player_style_instruction(style)
    logic_mode = style_logic_mode(style, style_instruction)
    properties = {
        "answer": {"type": "string"},
        "explanation": {"type": "string"},
        "try_it": {"type": "string"},
    }
    required = ["answer", "explanation", "try_it"]
    if logic_mode == "sarcasm":
        properties.update({
            "surface_meaning": {"type": "string"},
            "hidden_meaning": {"type": "string"},
        })
        required.extend(["surface_meaning", "hidden_meaning"])
    schema = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }

    base_prompt = f"""
あなたは、5〜6歳の子どもがカードゲーム「言いカエル」の発想のしかたを学ぶための先生役です。
完成回答をたくさん見せるのではなく、参考回答は1つだけ出してください。
そして「どうしてその言葉を思いついたのか」を、子どもが次に自分でまねできるように説明してください。

【お題】
{topic}

【言い方カード】
{style}

【この言い方カードの狙い】
{style_instruction}

{sarcasm_generation_rules() if logic_mode == "sarcasm" else ""}

【最重要】
- 参考回答は1つだけ。
- 答えの巧さよりも、発想のしかたを教えることを重視する。
- 一休さんのように、普通の見方を1回だけずらして「なるほど」と思えるウィットにする。
- 言い方カードは表面の語尾・擬音・派手な言葉ではなく、「どう見直すか」の条件として使う。
- お題カードの文言「{topic}」を answer にそのまま使わない。
- answer は原則ワンフレーズ、最大でも短いツーフレーズ。30文字以内を目安にする。
- 5〜6歳が頭に絵を浮かべられ、少し笑える内容にする。
- 難しい熟語は避ける。皮肉モードでは、5〜6歳にも説明できる「ほめているようで、ほんとうは少し逆」の二重の意味にする。
- 日本語の漢字・ひらがな・カタカナ・数字・一般的な句読点だけを使う。外国語文字やアルファベットは使わない。

【explanation の作り方】
- 内部の細かい思考手順を列挙するのではなく、子ども向けの短い学習解説にする。
- 2〜3文程度。
- 「お題の○○というところを見たよ。そこを△△みたいに見方を変えたよ。だから□□という言葉にしたよ。」のように、
  ①どの特徴を見たか、②どう見方を変えたか、③言い方カードにどう合わせたか、が分かるようにする。
- 「なぜ面白いか」も、子どもが分かる言葉で一言入れてよい。
- 皮肉モードでは「表では何をほめているように聞こえるか」と「ほんとうは何をちょっと困った点として見ているか」の両方を、子ども向けに説明する。

【try_it の作り方】
- 次に子ども自身が別の答えを作るためのコツを1文だけ。
- 完成回答をもう1つ出してはいけない。
- 例：「お題の特徴を、別のものの仕事に置きかえてみよう」のように、考え方だけを渡す。

【出力】
answer: 参考回答1つ
explanation: 子ども向けの発想解説
try_it: 自分で考えるためのコツ1つ
""".strip()

    def topic_is_reused(answer):
        answer_text = str(answer or "").replace(" ", "").replace("　", "")
        topic_text = str(topic or "").replace(" ", "").replace("　", "")
        return bool(topic_text and topic_text in answer_text)

    last_problems = []
    for attempt in range(4):
        retry_note = ""
        if attempt:
            retry_note = (
                "\n\n【作り直し】\n"
                + "、".join(last_problems)
                + "の条件に違反しました。答えは1つだけにし、日本語だけで、"
                "お題の言葉をそのまま使わず、子どもに発想のしかたが伝わる形へ作り直してください。"
            )
            if logic_mode == "sarcasm":
                retry_note += (
                    " 今回は皮肉なので、表ではほめているように聞こえ、"
                    "裏ではやりすぎ・困る点・矛盾を短く刺し、最後にフォローせず、answer単体で皮肉と分かるキレを残してください。"
                )

        result = ask_json(
            base_prompt + retry_note,
            "reference_answer",
            schema,
            max_output_tokens=520,
        )
        last_problems = []
        if topic_is_reused(result.get("answer", "")):
            last_problems.append("お題の文言をそのまま使用")
        if any(non_japanese_letters(result.get(key, "")) for key in ("answer", "explanation", "try_it")):
            last_problems.append("日本語以外の文字を使用")
        if not str(result.get("answer", "")).strip():
            last_problems.append("参考回答が空")
        if logic_mode == "sarcasm":
            if not str(result.get("surface_meaning", "")).strip() or not str(result.get("hidden_meaning", "")).strip():
                last_problems.append("皮肉の表と裏の意味が不足")
            if not last_problems:
                passed, reason = review_sarcasm_reference(topic, result)
                if not passed:
                    last_problems.append("皮肉判定:" + (reason or "二重の意味が弱い"))
        if not last_problems:
            return result

    raise ValueError("子ども向けの参考回答を作れませんでした。もう一度試してください。")


def reference_speech_text(item):
    return (
        f"参考回答は、{item['answer']}。"
        f"どう考えたかというと、{item['explanation']}。"
        f"自分で考えるコツは、{item['try_it']}。"
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
    "reference_answer": None,
    "reference_audio": None,
    "reference_audio_autoplay_pending": False,
    "topic_explanation": None,
    "topic_explanation_audio": None,
    "topic_explanation_autoplay_pending": False,
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


def preset_card_selection(field_key, audio_key, selected_text):
    """Preset one private Supabase card as if it had been selected from voice candidates."""
    transcript_key = f"_{field_key}_transcript"
    candidates_key = f"_{field_key}_candidates"
    digest_key = f"_{audio_key}_digest"
    radio_key = f"_{field_key}_radio"
    manual_key = f"_{field_key}_manual"

    # Clear a prior recorder value so an old recording does not overwrite the random draw.
    st.session_state.pop(audio_key, None)
    st.session_state.pop(digest_key, None)
    st.session_state.pop(manual_key, None)
    st.session_state[transcript_key] = "ランダムで選びました"
    st.session_state[candidates_key] = [selected_text]
    st.session_state[radio_key] = selected_text


def choose_random_round_cards(round_serial):
    rows = load_card_rows()
    topics = [
        row["card_text"]
        for row in rows
        if row["card_type"] == "topic" and not row.get("is_free_topic", False)
    ]
    styles = [row["card_text"] for row in rows if row["card_type"] == "style"]
    if not topics or not styles:
        raise ValueError("ランダムに選べるカードがありません。")

    topic = random.choice(topics)
    style = random.choice(styles)
    preset_card_selection(
        f"topic_draft_{round_serial}",
        f"topic_audio_{round_serial}",
        topic,
    )
    preset_card_selection(
        f"style_draft_{round_serial}",
        f"style_audio_{round_serial}",
        style,
    )
    # ANY自由入力の残骸があれば消す。
    st.session_state.pop(f"any_topic_{round_serial}", None)
    return topic, style


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
    if transcript == "ランダムで選びました":
        st.caption("🎲 ランダムで選びました。")
    elif transcript:
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
st.caption("お題・言い方の解説、参考回答、AI審判、AIプレイヤーの4つの使い方ができます。")
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
    st.caption("声で実物のカードを選ぶか、Supabaseの非公開カード一覧からランダムに1組選べます。")

    if st.button(
        "🎲 お題と言い方をランダムに選ぶ",
        use_container_width=True,
        key=f"random_cards_{st.session_state.round_serial}",
    ):
        try:
            choose_random_round_cards(st.session_state.round_serial)
            st.rerun()
        except Exception as exc:
            st.error("カードをランダムに選べませんでした。")
            with st.expander("保護者向け詳細"):
                st.code(str(exc))

    st.caption("または、カードを声で読むと、音の近い実在カードだけを候補表示します。")
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
    if is_free_topic_card(topic):
        topic = st.text_input(
            "ANYのお題",
            placeholder="出題者が考えた今回のお題",
            key=f"any_topic_{st.session_state.round_serial}",
        ).strip()
    ai_join = st.checkbox("④ AIもこのラウンドに参加する", value=True)

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
st.caption("お題と言い方カードの意味が分からないときに使います。ボタンを押すだけで、両方の説明が子ども向けの音声で流れます。ゲームの完成回答は出しません。")

if st.button("🔊 お題と言い方を解説して！", use_container_width=True):
    try:
        with st.spinner("お題と言い方を分かりやすく説明しています…"):
            item = explain_topic_and_style_for_child(
                st.session_state.topic,
                st.session_state.style,
            )
            audio_text = topic_explanation_speech_text(item)
            if not audio_text:
                raise ValueError("お題と言い方の解説が空でした。")
            st.session_state.topic_explanation = item
            st.session_state.topic_explanation_audio = speech_bytes(audio_text)
            st.session_state.topic_explanation_autoplay_pending = True
            log_topic_explanation(item)
        st.rerun()
    except Exception as exc:
        st.error("お題と言い方を解説できませんでした。もう一度試してください。")
        with st.expander("保護者向け詳細"):
            st.code(str(exc))

if st.session_state.topic_explanation:
    item = st.session_state.topic_explanation
    st.markdown(
        f"""
        <div class="support-card">
          <b>お題のせつめい</b><br><br>
          {item.get('topic_meaning', '')}<br><br>
          <b>言い方のせつめい</b><br><br>
          {item.get('style_meaning', '')}
        </div>
        """,
        unsafe_allow_html=True,
    )

if st.session_state.topic_explanation_audio:
    st.audio(
        st.session_state.topic_explanation_audio,
        format="audio/wav",
        autoplay=bool(st.session_state.topic_explanation_autoplay_pending),
    )
    st.session_state.topic_explanation_autoplay_pending = False


st.divider()


# -------------------- AI reference-answer mode --------------------
st.subheader("② AIの参考回答")
st.caption(
    "参考回答は1つだけ。答えそのものより、『どこを見て、どう見方を変えたか』を子ども向けに説明します。"
    "④のAIプレイヤーとは別に生成するため、AIプレイヤーの伏せ回答は見えません。"
)

if st.session_state.reference_answer is None:
    if st.button("💡 AIの参考回答を見る", use_container_width=True):
        try:
            with st.spinner("答えと考え方を1つ作っています…"):
                st.session_state.reference_answer = reference_answer(
                    st.session_state.topic,
                    st.session_state.style,
                )
            st.session_state.reference_audio = None
            st.session_state.reference_audio_autoplay_pending = False
            st.rerun()
        except Exception as exc:
            st.error("参考回答を作れませんでした。もう一度試してください。")
            with st.expander("保護者向け詳細"):
                st.code(str(exc))
else:
    item = st.session_state.reference_answer
    st.markdown(
        f"""
        <div class="answer-card">
          <div class="small-note">参考回答</div>
          <div class="answer-main">{item['answer']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="support-card">
          <b>どう考えたの？</b><br><br>
          {item['explanation']}<br><br>
          <b>自分で考えるコツ</b><br>
          {item['try_it']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔊 答えと考え方を聞く", use_container_width=True):
            try:
                with st.spinner("声を作っています…"):
                    st.session_state.reference_audio = speech_bytes(
                        reference_speech_text(item)
                    )
                st.session_state.reference_audio_autoplay_pending = True
                st.rerun()
            except Exception as exc:
                st.error("読み上げ音声を作れませんでした。")
                with st.expander("保護者向け詳細"):
                    st.code(str(exc))
    with c2:
        if st.button("↻ 別の参考回答", use_container_width=True):
            try:
                with st.spinner("別の見方を1つ考えています…"):
                    st.session_state.reference_answer = reference_answer(
                        st.session_state.topic,
                        st.session_state.style,
                    )
                st.session_state.reference_audio = None
                st.session_state.reference_audio_autoplay_pending = False
                st.rerun()
            except Exception as exc:
                st.error("参考回答を作れませんでした。もう一度試してください。")
                with st.expander("保護者向け詳細"):
                    st.code(str(exc))

    if st.session_state.reference_audio:
        st.audio(
            st.session_state.reference_audio,
            format="audio/wav",
            autoplay=bool(st.session_state.reference_audio_autoplay_pending),
        )
        st.session_state.reference_audio_autoplay_pending = False

st.divider()


# -------------------- Judging mode --------------------
st.subheader("③ AI審判・採点モード")
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
st.subheader("④ AIもゲームに参加")
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
