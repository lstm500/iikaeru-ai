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
from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo

import streamlit as st
from openai import OpenAI

try:
    from supabase import create_client
except Exception:
    create_client = None


# ============================================================
# Basic settings
# v29: two AI frogs; crayon/dessin image generation is parallelized for faster display.
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


def generate_player_image(topic, style, label, answer, why, art_style="crayon"):
    art_instruction = {
        "crayon": "Use a child-friendly crayon drawing style with thick waxy strokes, bright colors, and a playful picture-book feeling.",
        "dessin": "Use a child-friendly colored pencil sketch / dessin style with hand-drawn shading, soft lines, and a lightly realistic but warm look.",
    }.get(art_style, "Use a child-friendly picture-book illustration style.")
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
- {art_instruction}
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


def generate_dual_images(topic, style, label, answer, why):
    """Generate crayon and dessin images in parallel to reduce wait time."""
    jobs = {
        "crayon": ("crayon",),
        "dessin": ("dessin",),
    }
    images = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            key: executor.submit(
                generate_player_image,
                topic,
                style,
                label,
                answer,
                why,
                art_style,
            )
            for key, (art_style,) in jobs.items()
        }
        for key, future in futures.items():
            images[key] = future.result()
    return images


def generate_reference_images(topic, style, answer, explanation):
    return generate_dual_images(topic, style, "参考回答", answer, explanation)


def generate_reference_assets(topic, style, item):
    """Generate the two pictures and narration concurrently."""
    with ThreadPoolExecutor(max_workers=3) as executor:
        f_crayon = executor.submit(
            generate_player_image, topic, style, "参考回答", item["answer"], item["explanation"], "crayon"
        )
        f_dessin = executor.submit(
            generate_player_image, topic, style, "参考回答", item["answer"], item["explanation"], "dessin"
        )
        f_audio = executor.submit(speech_bytes, reference_speech_text(item))
        images = {"crayon": f_crayon.result(), "dessin": f_dessin.result()}
        audio = f_audio.result()
    return images, audio


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
    topic_instruction = player_topic_instruction(topic)
    base_prompt = f"""
あなたは、5〜6歳の子どもにカードゲーム「言いカエル」のカードの意味をやさしく説明する先生役です。

【今回のお題カード】
{topic}

【お題カードの一般的なとらえ方】
{topic_instruction}

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
- topic_meaning は「お題カードの一般的なとらえ方」を優先する。珍しい例外や、一般的な意味と逆の説明を標準として扱わない。
- 「一般的なとらえ方」は絶対の決めつけではなく、ゲームで共有しやすい定番イメージとして使う。
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


def player_topic_instruction(topic):
    """Load the conventional/common interpretation of a topic from the private card master."""
    target = str(topic or "").strip()
    for row in load_card_rows():
        if row["card_type"] == "topic" and row["card_text"] == target:
            instruction = str(row.get("ai_instruction") or "").strip()
            if instruction:
                return instruction
    return (
        "このお題について、日本で一般的に共有される意味・典型的な場面・定番のイメージを基準にする。"
        "珍しい例外や言葉遊びだけを標準的な意味として扱わない。"
        "逆転や皮肉を作る場合も、事実関係を逆にせず、評価や見方だけをひねる。"
    )


def style_logic_mode(style, style_instruction):
    """Choose a generation strategy from the selected style and its private guidance."""
    text = f"{style}\n{style_instruction}"
    # Special styles get their own reasoning + review path so surface decoration
    # cannot overpower the actual meaning of the card.
    if "皮肉" in text or ("ほめ" in text and ("本当の意味" in text or "裏" in text)):
        return "sarcasm"
    if "詩的" in text or ("情景" in text and "比喩" in text):
        return "poetic"
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


def poetic_generation_rules():
    return """
【詩的に専用ロジック】
「きれいな言葉」を足すだけでは詩的とはしません。お題から比喩が飛びすぎないことを最優先にします。

【必ず守る3段階】
1. お題の一般的なとらえ方から、中心的で具体的な特徴を1つだけ選ぶ。
2. その特徴と「同じところ」がある身近な物・自然・場面を1つだけ選ぶ。
3. その共通点を橋にして、短い情景や比喩にする。

【比喩の橋テスト】
内部で必ず「お題の○○と、たとえた△△は、□□というところが同じ」と1文で説明する。
この1文が自然に作れない比喩は捨てる。
共通点は、動き・形・役割・時間・音・温度・重さ・明るさ・集まり方など、子どもにも分かる具体的なものにする。

【ずれを防ぐルール】
- お題と関係のない「月・星・風・光・夢・空・海・花」などを、きれいだからという理由だけで足さない。
- たとえを二段三段と連鎖させない。比喩は原則1つ。
- お題の定番イメージにない出来事を勝手に作らない。
- 比喩先の物語が主役になり、お題の特徴が見えなくなったら不合格。
- answerだけでは少し余韻があってよいが、whyを読むと「どこが似ているか」が一発で分かること。
- 詩的さは、関係の薄い美辞麗句ではなく「ぴったりした一枚の情景」から作る。

【よい詩的表現の条件】
- お題の特徴が先、比喩は後。
- 意外だが、共通点を聞くと「なるほど」と戻ってこられる。
- 5〜6歳が具体的な絵を思い浮かべられる。
- 短く、比喩が1枚の絵としてまとまっている。

内部では少なくとも10案を作り、「お題との近さ」「共通点の明確さ」「詩的な余韻」「子どもの分かりやすさ」を比べて最も高い案を残す。
""".strip()


def review_poetic_player_answers(topic, topic_instruction, result):
    """Reject poetic answers whose metaphor has drifted away from the topic anchor."""
    review_item = {
        "type": "object",
        "properties": {
            "pass": {"type": "boolean"},
            "anchor_score": {"type": "integer", "minimum": 0, "maximum": 3},
            "bridge_score": {"type": "integer", "minimum": 0, "maximum": 3},
            "poetic_score": {"type": "integer", "minimum": 0, "maximum": 3},
            "child_score": {"type": "integer", "minimum": 0, "maximum": 3},
            "topic_anchor": {"type": "string"},
            "metaphor_target": {"type": "string"},
            "shared_feature": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": [
            "pass", "anchor_score", "bridge_score", "poetic_score", "child_score",
            "topic_anchor", "metaphor_target", "shared_feature", "reason"
        ],
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
            "why": str((result.get(key, {}) or {}).get("why", "")),
        }
        for key in ("metaphor", "nickname", "twist")
    }
    review = ask_json(
        f"""
あなたは子ども向け言葉ゲームの厳しい「詩的な比喩のずれ判定係」です。
美しさより先に、お題と比喩のつながりを確認してください。

【お題】
{topic}

【お題の一般的なとらえ方】
{topic_instruction}

【回答】
{json.dumps(compact, ensure_ascii=False)}

各回答について、次を0〜3点で判定してください。
- anchor_score：お題の一般的な特徴を、具体的に1つつかんでいるか。
- bridge_score：「お題の○○と、たとえた△△は、□□が同じ」と自然な1文で結べるか。
- poetic_score：関係のない美辞麗句ではなく、ぴったりした一枚の情景になっているか。
- child_score：5〜6歳が説明を聞けば場面を思い浮かべられるか。

厳しい不合格条件：
- 月、星、風、光、夢、空、海、花などを、きれいだからというだけで足している。
- お題の中心的特徴と比喩先の共通点が曖昧。
- 比喩が二段以上に連鎖し、何をたとえているか分からない。
- answerやwhyの説明を読んでも、お題へ自然に戻れない。
- 比喩先の物語が主役になり、お題が置き去りになっている。

比喩が意外でも、shared_featureが具体的で筋が通れば合格してよいです。
pass=true は4項目すべて2点以上、かつ bridge_score が2点以上の場合だけにしてください。
reasonは短い日本語1文にしてください。
""".strip(),
        "poetic_review",
        schema,
        max_output_tokens=700,
    )
    problems = []
    for key in ("metaphor", "nickname", "twist"):
        item = review.get(key, {}) or {}
        scores_ok = all(
            int(item.get(score, 0) or 0) >= 2
            for score in ("anchor_score", "bridge_score", "poetic_score", "child_score")
        )
        if not item.get("pass", False) or not scores_ok:
            reason = str(item.get("reason", "お題と比喩の橋が弱い")).strip()
            problems.append(f"{key}:{reason}")
    return problems


def review_poetic_reference(topic, topic_instruction, result):
    schema = {
        "type": "object",
        "properties": {
            "pass": {"type": "boolean"},
            "anchor_score": {"type": "integer", "minimum": 0, "maximum": 3},
            "bridge_score": {"type": "integer", "minimum": 0, "maximum": 3},
            "poetic_score": {"type": "integer", "minimum": 0, "maximum": 3},
            "child_score": {"type": "integer", "minimum": 0, "maximum": 3},
            "topic_anchor": {"type": "string"},
            "metaphor_target": {"type": "string"},
            "shared_feature": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": [
            "pass", "anchor_score", "bridge_score", "poetic_score", "child_score",
            "topic_anchor", "metaphor_target", "shared_feature", "reason"
        ],
        "additionalProperties": False,
    }
    review = ask_json(
        f"""
あなたは「詩的に」の参考回答を判定する係です。

【お題】{topic}
【一般的なとらえ方】{topic_instruction}
【参考回答】{result.get('answer', '')}
【子ども向け説明】{result.get('explanation', '')}

必ず「お題の○○と、たとえた△△は、□□というところが同じ」と1文で結べるか確認してください。
きれいな単語が入っていても、その共通点が弱ければ不合格です。
比喩は原則1つ。お題の特徴が主役で、比喩はそれを見せるための道具になっている必要があります。

0〜3点：
- anchor_score：お題の具体的特徴をつかんでいる。
- bridge_score：お題と比喩の共通点が具体的で自然。
- poetic_score：一枚の情景として詩的。
- child_score：5〜6歳にも説明できる。

pass=true は4項目すべて2点以上の場合だけです。
""".strip(),
        "poetic_reference_review",
        schema,
        max_output_tokens=330,
    )
    scores_ok = all(
        int(review.get(score, 0) or 0) >= 2
        for score in ("anchor_score", "bridge_score", "poetic_score", "child_score")
    )
    return bool(review.get("pass", False) and scores_ok), str(review.get("reason", "")).strip()


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


def review_topic_alignment(topic, topic_instruction, style, answer_items):
    """Check that creative answers stay anchored to the conventional meaning of the topic."""
    schema = {
        "type": "object",
        "properties": {
            "passed": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["passed", "reason"],
        "additionalProperties": False,
    }
    joined = "\n".join(
        f"{label}: {item.get('answer', '')} / 理由: {item.get('why', item.get('explanation', ''))}"
        for label, item in answer_items
    )
    return ask_json(
        f"""
あなたはカードゲーム『言いカエル』の意味確認係です。
創作の面白さではなく、お題について一般に共有される意味・定番イメージから不自然に外れていないかだけを確認してください。

【お題】
{topic}

【一般的なとらえ方】
{topic_instruction}

【言い方カード】
{style}

【確認する回答】
{joined}

【判定基準】
- 一般的なとらえ方にある中心的な事実・役割・状況を土台にしていれば合格。
- 比喩、誇張、皮肉、逆転、あだ名は自由。ただし「事実そのもの」を逆にしてはいけない。
- 逆転は「困る→役立つ」「弱点→長所」など評価や見方を反転するのはよいが、「本来あるものを無いことにする」「普通は起きることを起きないことにする」など、定番の前提を打ち消すだけの回答は不合格。
- 珍しい例外を、あたかもそのお題の普通の姿のように扱う回答は不合格。
- 一般的なとらえ方に複数の側面がある場合は、そのどれか1つに正しく乗っていればよい。
- 人については、性格・能力・健康状態を見た目や属性だけから決めつけない。

問題なければ passed=true。ずれていれば passed=false にして、reason に何が一般認識と食い違うかを短く書いてください。
""".strip(),
        "topic_alignment_review",
        schema,
        max_output_tokens=220,
    )


def player_answers(topic, style):
    style_instruction = player_style_instruction(style)
    topic_instruction = player_topic_instruction(topic)
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

【このお題の一般的な意味・定番イメージ】
{topic_instruction}

【言い方カード】
{style}

【この言い方カードの意味・狙い】
{style_instruction}

{sarcasm_generation_rules() if logic_mode == "sarcasm" else poetic_generation_rules() if logic_mode == "poetic" else ""}

【お題の扱い方】
- まず「このお題の一般的な意味・定番イメージ」を土台にする。珍しい例外や、一般的な意味と逆の前提から出発しない。
- ひねるのは「評価・見方・役割・たとえ」であって、中心的な事実関係そのものではない。
- ぎゃくてんカエルでも、事実を反対にするのではなく、「大変→役に立つ」「困る→面白い」のように価値づけを反転する。
- お題に複数の定番イメージがあるときは、そのうち1つを明確に選んで発想の土台にする。

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
- 「詩的に」の場合は、お題の具体的特徴を1つ先に固定し、その特徴と共通点が1文で説明できる比喩だけを使う。美しい単語から逆算してお題へ近づける作り方は禁止。

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
            elif logic_mode == "poetic":
                retry_note += (
                    " 今回は詩的表現なので、きれいな単語を増やすのではなく、"
                    "お題の具体的特徴を1つ選び、『お題の特徴と比喩先の共通点』を1文で言える比喩へ戻してください。"
                    "月・星・風・光などを関係なく足さず、比喩は1つの情景に絞ってください。"
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

        if not last_problems:
            alignment = review_topic_alignment(
                topic,
                topic_instruction,
                style,
                [
                    ("たとえカエル", result.get("metaphor", {}) or {}),
                    ("なまえカエル", result.get("nickname", {}) or {}),
                    ("ぎゃくてんカエル", result.get("twist", {}) or {}),
                ],
            )
            if not alignment.get("passed", False):
                last_problems.append("お題の一般認識とのずれ:" + str(alignment.get("reason", "")))

        if logic_mode == "sarcasm":
            for key in ("metaphor", "nickname", "twist"):
                item = result.get(key, {}) or {}
                if not str(item.get("surface_meaning", "")).strip() or not str(item.get("hidden_meaning", "")).strip():
                    last_problems.append(f"{key}:表と裏の意味が不足")
            if not last_problems:
                last_problems.extend(review_sarcasm_player_answers(topic, result))
        elif logic_mode == "poetic" and not last_problems:
            last_problems.extend(review_poetic_player_answers(topic, topic_instruction, result))

        if not last_problems:
            return result

    raise ValueError("日本語だけで、お題の文言をそのまま使わない回答を作れませんでした。もう一度AIの回答を生成してください。")


def ai_game_answer(topic, style):
    """Create one playful AI answer for game participation."""
    style_instruction = player_style_instruction(style)
    topic_instruction = player_topic_instruction(topic)
    logic_mode = style_logic_mode(style, style_instruction)
    properties = {
        "answer": {"type": "string"},
        "why": {"type": "string"},
    }
    required = ["answer", "why"]
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
あなたは、5〜6歳の子どもと親が遊ぶカードゲーム「言いカエル」に参加するAIです。
AIは1人ぶんの回答を1つだけ出します。
最優先は、子どもが聞いてすぐ場面を想像できて、クスッと笑えることです。
ただし、意味不明な奇抜さではなく、お題と言い方カードにちゃんと沿っている必要があります。

【お題】
{topic}

【このお題の一般的な意味・定番イメージ】
{topic_instruction}

【言い方カード】
{style}

【この言い方カードの意味・狙い】
{style_instruction}

{sarcasm_generation_rules() if logic_mode == "sarcasm" else poetic_generation_rules() if logic_mode == "poetic" else ""}

【方針】
- まず、お題の一般的なイメージを土台にする。
- そのうえで、見方・役割・たとえ・場面のどれかを1回だけずらして面白くする。
- 子どもにとって、頭に絵が浮かぶ具体性を大切にする。
- 「言い方カードに沿っていること」と「子どもに面白いこと」の両方を満たす。
- ただ語尾を変えるだけ、ただ派手な言葉を足すだけ、ただ称号をつけるだけは禁止。
- お題カードの文言「{topic}」を answer にそのまま使わない。
- answer はワンフレーズ、または短いツーフレーズ。30文字以内を目安にする。
- why は子ども向けに1〜2文で、「どこが面白いか」「どこをどう見方を変えたか」を簡単に説明する。
- 皮肉のときは、やさしすぎず、ほめている形の中に軽いトゲとキレを残す。ただし子ども向けなので過度にきつくしない。
- 詩的のときは、お題と比喩先の共通点がちゃんと分かるようにする。
- 日本語以外の文字は使わない。

【内部で比較すること】
- 少なくとも8案を内部で考え、最も「子どもが笑える」「言い方カードに合う」「お題からずれていない」案を1つ選ぶ。
- 面白さを優先してよいが、意味不明なナンセンスは不可。

【出力】
answer: AIの回答1つ
why: 子ども向けの短い解説
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
                + "の条件に違反しました。もっと子どもが笑いやすく、"
                "ただしお題と言い方カードからはずれない1回答へ作り直してください。"
            )
        result = ask_json(
            base_prompt + retry_note,
            "ai_game_answer",
            schema,
            max_output_tokens=360,
        )
        last_problems = []
        answer_text = str(result.get("answer", "")).strip()
        why_text = str(result.get("why", "")).strip()
        if not answer_text:
            last_problems.append("回答が空")
        if not why_text:
            last_problems.append("解説が空")
        if topic_is_reused(answer_text):
            last_problems.append("お題の文言をそのまま使用")
        if any(non_japanese_letters(result.get(key, "")) for key in ("answer", "why")):
            last_problems.append("日本語以外の文字を使用")
        if not last_problems:
            alignment = review_topic_alignment(
                topic,
                topic_instruction,
                style,
                [("AI回答", {"answer": answer_text, "why": why_text})],
            )
            if not alignment.get("passed", False):
                last_problems.append("お題の一般認識とのずれ:" + str(alignment.get("reason", "")))
        if logic_mode == "sarcasm" and not last_problems:
            if not str(result.get("surface_meaning", "")).strip() or not str(result.get("hidden_meaning", "")).strip():
                last_problems.append("皮肉の表と裏の意味が不足")
        elif logic_mode == "poetic" and not last_problems:
            passed, reason = review_poetic_reference(topic, topic_instruction, {
                "answer": answer_text,
                "explanation": why_text,
            })
            if not passed:
                last_problems.append("詩的比喩判定:" + (reason or "お題と比喩の共通点が弱い"))
        if not last_problems:
            return result
    raise ValueError("AIの回答を作れませんでした。もう一度試してください。")


def ai_game_answers(topic, style):
    """Create two distinct, child-funny AI answers in one text request."""
    style_instruction = player_style_instruction(style)
    topic_instruction = player_topic_instruction(topic)
    logic_mode = style_logic_mode(style, style_instruction)

    item_properties = {
        "answer": {"type": "string"},
        "why": {"type": "string"},
    }
    item_required = ["answer", "why"]
    if logic_mode == "sarcasm":
        item_properties.update({
            "surface_meaning": {"type": "string"},
            "hidden_meaning": {"type": "string"},
        })
        item_required.extend(["surface_meaning", "hidden_meaning"])

    item_schema = {
        "type": "object",
        "properties": item_properties,
        "required": item_required,
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {
            "frog1": item_schema,
            "frog2": item_schema,
        },
        "required": ["frog1", "frog2"],
        "additionalProperties": False,
    }

    prompt = f"""
あなたは、5〜6歳の子どもと親が遊ぶカードゲーム「言いカエル」に参加するAIです。
AIカエルは2匹いますが、カエルに名前は付けません。
2匹がそれぞれ1つずつ、合計2つの別々の回答を出してください。

【お題】
{topic}

【このお題の一般的な意味・定番イメージ】
{topic_instruction}

【言い方カード】
{style}

【この言い方カードの意味・狙い】
{style_instruction}

{sarcasm_generation_rules() if logic_mode == "sarcasm" else poetic_generation_rules() if logic_mode == "poetic" else ""}

【最優先】
- 5〜6歳の子どもが聞いて、場面をすぐ想像できて、クスッと笑える回答を優先する。
- ただし、お題の一般的な意味と言い方カードから外れない。
- 2匹の回答は発想の方向を変え、似た答えにしない。
- 意味不明なナンセンスではなく、「そう来たか」と分かる面白さにする。
- 見方・役割・たとえ・場面のどれかを1回だけずらす。
- 語尾だけ、擬音だけ、派手な称号だけで面白く見せるのは禁止。
- お題カードの文言「{topic}」を answer にそのまま使わない。
- answer はワンフレーズまたは短いツーフレーズ、30文字以内を目安にする。
- why は子ども向けに1〜2文で、どこが面白いのかを簡単に説明する。
- 皮肉では、ほめている形の中に軽いトゲとキレを残す。ただし人格攻撃はしない。
- 詩的では、お題と比喩先の共通点が子どもにも分かるようにする。
- 日本語以外の文字を使わない。

【内部での選び方】
1匹につき少なくとも6案を内部で考え、
「子どもの面白さ」「お題との一致」「言い方カードとの一致」「分かりやすさ」が高い案を1つずつ選ぶ。

【出力】
frog1: 1匹目の回答と解説
frog2: 2匹目の回答と解説
""".strip()

    def topic_is_reused(answer):
        answer_text = str(answer or "").replace(" ", "").replace("　", "")
        topic_text = str(topic or "").replace(" ", "").replace("　", "")
        return bool(topic_text and topic_text in answer_text)

    last_problems = []
    for attempt in range(4):
        retry = ""
        if attempt:
            retry = (
                "\n\n【作り直し】\n"
                + "、".join(last_problems)
                + "の条件に違反しました。2匹の発想をはっきり変え、もっと子どもが笑いやすく、"
                "それでもお題と言い方カードに沿う回答へ作り直してください。"
            )
        result = ask_json(prompt + retry, "ai_game_answers_two", schema, max_output_tokens=620)
        problems = []
        items = [result.get("frog1", {}) or {}, result.get("frog2", {}) or {}]
        answers = [str(x.get("answer", "")).strip() for x in items]
        if any(not a for a in answers):
            problems.append("回答が空")
        if answers[0] and answers[1] and answers[0] == answers[1]:
            problems.append("2匹の回答が同じ")
        if any(topic_is_reused(a) for a in answers):
            problems.append("お題の文言をそのまま使用")
        for item in items:
            if non_japanese_letters(item.get("answer", "")) or non_japanese_letters(item.get("why", "")):
                problems.append("日本語以外の文字を使用")
                break
        if not problems:
            alignment = review_topic_alignment(
                topic,
                topic_instruction,
                style,
                [
                    ("1匹目", items[0]),
                    ("2匹目", items[1]),
                ],
            )
            if not alignment.get("passed", False):
                problems.append("お題の一般認識とのずれ:" + str(alignment.get("reason", "")))
        if logic_mode == "sarcasm" and not problems:
            for i, item in enumerate(items, start=1):
                if not str(item.get("surface_meaning", "")).strip() or not str(item.get("hidden_meaning", "")).strip():
                    problems.append(f"{i}匹目の皮肉の表と裏の意味が不足")
        elif logic_mode == "poetic" and not problems:
            for i, item in enumerate(items, start=1):
                passed, reason = review_poetic_reference(
                    topic,
                    topic_instruction,
                    {"answer": item.get("answer", ""), "explanation": item.get("why", "")},
                )
                if not passed:
                    problems.append(f"{i}匹目の詩的比喩判定:" + (reason or "共通点が弱い"))
        if not problems:
            return result
        last_problems = problems

    raise ValueError("2匹のAI回答を作れませんでした。もう一度試してください。")


def reference_answer(topic, style):
    """Create one pedagogical reference answer with a child-friendly explanation."""
    style_instruction = player_style_instruction(style)
    topic_instruction = player_topic_instruction(topic)
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

【このお題の一般的な意味・定番イメージ】
{topic_instruction}

【言い方カード】
{style}

【この言い方カードの狙い】
{style_instruction}

{sarcasm_generation_rules() if logic_mode == "sarcasm" else poetic_generation_rules() if logic_mode == "poetic" else ""}

【お題の扱い】
- 「このお題の一般的な意味・定番イメージ」を必ず発想の出発点にする。
- ひねるのは見方・価値づけ・たとえであり、定番の事実そのものを反対にしない。
- 逆転や皮肉でも、一般的な前提を打ち消して別物にするのではなく、その前提を残したまま見え方を変える。

【最重要】
- 参考回答は1つだけ。
- 答えの巧さよりも、発想のしかたを教えることを重視する。
- 一休さんのように、普通の見方を1回だけずらして「なるほど」と思えるウィットにする。
- 言い方カードは表面の語尾・擬音・派手な言葉ではなく、「どう見直すか」の条件として使う。
- お題カードの文言「{topic}」を answer にそのまま使わない。
- answer はごく短いワンフレーズ中心。長さは今までの三分の一程度として、6〜12文字くらい、長くても16文字以内を目安にする。
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
- 詩的モードでは「お題のどの特徴」と「何にたとえたか」と「どこが同じか」を必ず説明する。「○○と△△は、□□なところが似ているからだよ」と言える具体的な橋を見せる。

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
            )
            if logic_mode == "sarcasm":
                retry_note += (
                    " 今回は皮肉なので、表ではほめているように聞こえ、"
                    "裏ではやりすぎ・困る点・矛盾を短く刺し、最後にフォローせず、answer単体で皮肉と分かるキレを残してください。"
                )
            elif logic_mode == "poetic":
                retry_note += (
                    " 今回は詩的表現なので、お題の具体的特徴を1つ固定し、"
                    "その特徴と比喩先の共通点が子どもにも分かる1文になるように作り直してください。"
                    "きれいな単語を先に選ぶ作り方や、比喩の連鎖は禁止です。"
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
        answer_text = str(result.get("answer", "")).strip()
        if not answer_text:
            last_problems.append("参考回答が空")
        if len(answer_text.replace(" ", "").replace("　", "")) > 16:
            last_problems.append("参考回答が長すぎる")
        if not last_problems:
            alignment = review_topic_alignment(
                topic,
                topic_instruction,
                style,
                [("参考回答", {"answer": result.get("answer", ""), "explanation": result.get("explanation", "")})],
            )
            if not alignment.get("passed", False):
                last_problems.append("お題の一般認識とのずれ:" + str(alignment.get("reason", "")))
        if logic_mode == "sarcasm":
            if not str(result.get("surface_meaning", "")).strip() or not str(result.get("hidden_meaning", "")).strip():
                last_problems.append("皮肉の表と裏の意味が不足")
            if not last_problems:
                passed, reason = review_sarcasm_reference(topic, result)
                if not passed:
                    last_problems.append("皮肉判定:" + (reason or "二重の意味が弱い"))
        elif logic_mode == "poetic" and not last_problems:
            passed, reason = review_poetic_reference(topic, topic_instruction, result)
            if not passed:
                last_problems.append("詩的比喩判定:" + (reason or "お題と比喩の共通点が弱い"))
        if not last_problems:
            return result

    raise ValueError("子ども向けの参考回答を作れませんでした。もう一度試してください。")


def reference_speech_text(item):
    return (
        f"参考回答は、{item['answer']}。"
        f"どう考えたかを説明するね。{item['explanation']}。"
        f"自分で考えるコツは、{item['try_it']}。"
    )


def player_speech_text(answers):
    return (
        f"1匹目の答えは、{answers['frog1']['answer']}。"
        f"どうしてそう言ったかというと、{answers['frog1']['why']}。"
        f"2匹目の答えは、{answers['frog2']['answer']}。"
        f"どうしてそう言ったかというと、{answers['frog2']['why']}。"
    )


def generate_ai_images(answers):
    """Generate four pictures (2 frogs x 2 art styles) concurrently."""
    jobs = []
    for frog_key, label in (("frog1", "1匹目"), ("frog2", "2匹目")):
        item = answers[frog_key]
        for art_style in ("crayon", "dessin"):
            jobs.append((frog_key, art_style, label, item))

    image_map = {"frog1": {}, "frog2": {}}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for frog_key, art_style, label, item in jobs:
            future = executor.submit(
                generate_player_image,
                st.session_state.topic,
                st.session_state.style,
                label,
                str(item.get("answer", "")),
                str(item.get("why", "")),
                art_style,
            )
            futures[(frog_key, art_style)] = future
        for (frog_key, art_style), future in futures.items():
            image_map[frog_key][art_style] = future.result()
    return image_map


def generate_ai_assets(answers):
    """Generate all four pictures and narration concurrently."""
    with ThreadPoolExecutor(max_workers=5) as executor:
        image_futures = {}
        for frog_key, label in (("frog1", "1匹目"), ("frog2", "2匹目")):
            item = answers[frog_key]
            for art_style in ("crayon", "dessin"):
                image_futures[(frog_key, art_style)] = executor.submit(
                    generate_player_image,
                    st.session_state.topic,
                    st.session_state.style,
                    label,
                    str(item.get("answer", "")),
                    str(item.get("why", "")),
                    art_style,
                )
        audio_future = executor.submit(speech_bytes, player_speech_text(answers))
        image_map = {"frog1": {}, "frog2": {}}
        for (frog_key, art_style), future in image_futures.items():
            image_map[frog_key][art_style] = future.result()
        audio = audio_future.result()
    return image_map, audio


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
    "reference_images": {},
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


def render_dual_images(images, prefix=""):
    if not images:
        return
    col1, col2 = st.columns(2)
    with col1:
        if images.get("crayon"):
            st.image(images["crayon"], caption=f"{prefix}クレヨン調", use_container_width=True)
    with col2:
        if images.get("dessin"):
            st.image(images["dessin"], caption=f"{prefix}デッサン調", use_container_width=True)


def render_ai_answers(answers, images):
    for frog_key, index in (("frog1", 1), ("frog2", 2)):
        item = answers[frog_key]
        st.markdown(f"**🐸 {index}匹目**")
        render_dual_images(images.get(frog_key, {}), prefix="")
        st.markdown(
            f"""
            <div class="answer-card">
              <div class="answer-main">{item['answer']}</div>
              <div class="small-note">{item['why']}</div>
            </div>
            """,
            unsafe_allow_html=True,
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
    "参考回答は1つだけ。クレヨン調とデッサン調の絵を先に2つ出し、そのあと答えと考え方を音声でやさしく説明します。"
    "④のAI参加とは別に生成するため、④の答えは見えません。"
)

if st.session_state.reference_answer is None:
    if st.button("💡 AIの参考回答を見る", use_container_width=True):
        try:
            with st.spinner("絵と答えを作っています…"):
                item = reference_answer(
                    st.session_state.topic,
                    st.session_state.style,
                )
                image_bytes, audio_bytes = generate_reference_assets(
                    st.session_state.topic,
                    st.session_state.style,
                    item,
                )
            st.session_state.reference_answer = item
            st.session_state.reference_images = image_bytes
            st.session_state.reference_audio = audio_bytes
            st.session_state.reference_audio_autoplay_pending = True
            st.rerun()
        except Exception as exc:
            st.error("参考回答を作れませんでした。もう一度試してください。")
            with st.expander("保護者向け詳細"):
                st.code(str(exc))
else:
    item = st.session_state.reference_answer
    render_dual_images(st.session_state.reference_images, prefix="")
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
        if st.button("🔊 もう一度説明を聞く", use_container_width=True):
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
                    item = reference_answer(
                        st.session_state.topic,
                        st.session_state.style,
                    )
                    image_bytes, audio_bytes = generate_reference_assets(
                        st.session_state.topic,
                        st.session_state.style,
                        item,
                    )
                st.session_state.reference_answer = item
                st.session_state.reference_images = image_bytes
                st.session_state.reference_audio = audio_bytes
                st.session_state.reference_audio_autoplay_pending = True
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
st.caption("AIカエルは2匹。名前は付けません。2匹とも子どもがクスッと笑える面白さを重視し、それぞれクレヨン調とデッサン調の絵を横並びで表示します。画像は4枚を並列生成して待ち時間を短くしています。")

# Hot-reload compatibility: reset only the AI section if an older one-frog format remains in session.
if st.session_state.ai_joined and st.session_state.ai_answers:
    current_ai = st.session_state.ai_answers
    if not (isinstance(current_ai, dict) and "frog1" in current_ai and "frog2" in current_ai):
        st.session_state.ai_joined = False
        st.session_state.ai_answers = None
        st.session_state.ai_revealed = False
        st.session_state.ai_images = {}
        st.session_state.ai_audio = None
        st.session_state.ai_audio_autoplay_pending = False

if not st.session_state.ai_joined:
    if st.button("AIカエル2匹もこのラウンドに参加", use_container_width=True):
        try:
            with st.spinner("2匹がこっそり考えています…"):
                answers = ai_game_answers(st.session_state.topic, st.session_state.style)
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
        st.info("AIカエル2匹も答えを考えました。まだ伏せています。")
        if st.button("AIカエル2匹の答えを見る", type="primary", use_container_width=True):
            try:
                with st.spinner("4枚の絵を同時に作っています…"):
                    answers = st.session_state.ai_answers
                    if answers is None:
                        answers = ai_game_answers(st.session_state.topic, st.session_state.style)
                        st.session_state.ai_answers = answers
                        update_round_history(
                            st.session_state.round_id,
                            ai_joined=True,
                            ai_answers=answers,
                        )
                    image_map, audio_bytes = generate_ai_assets(answers)
                st.session_state.ai_images = image_map
                st.session_state.ai_audio = audio_bytes
                st.session_state.ai_audio_autoplay_pending = True
                st.session_state.ai_revealed = True
                update_round_history(st.session_state.round_id, ai_revealed=True)
                st.rerun()
            except Exception as exc:
                st.error("AIの回答や絵を作れませんでした。もう一度試してください。")
                with st.expander("保護者向け詳細"):
                    st.code(str(exc))
    else:
        render_ai_answers(st.session_state.ai_answers, st.session_state.ai_images)
        if st.button("🔊 もう一度AIの説明を聞く", use_container_width=True):
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
        if st.session_state.ai_audio:
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
