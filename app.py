import io
import json
import os
import tempfile
import time
import hmac
import wave
import audioop
import hashlib
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
FAMILY_PIN = str(secret("FAMILY_PIN", "")).strip()
APP_TIMEZONE = secret("APP_TIMEZONE", "Asia/Tokyo")
SUPABASE_URL = secret("SUPABASE_URL", "")
SUPABASE_SECRET_KEY = secret("SUPABASE_SECRET_KEY", "")


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
    result = openai_client().responses.create(
        model=TEXT_MODEL,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": name,
                "strict": True,
                "schema": schema,
            }
        },
        max_output_tokens=max_output_tokens,
        store=False,
    )
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


def player_answers(topic, style):
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

【言い方】
{style}

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
1. 5〜6歳が聞いて意味や場面を想像できる。
2. 「なるほど」と「ちょっと変で面白い」が両方ある。
3. 既知の言葉を中心にしつつ、3回答のうち最大1つだけ少し新しい語彙を混ぜてよい。

【ルール】
- 3回答は発想を重複させない。
- 1回答は原則2〜12文字程度。長くても短い一言にする。
- 難解な熟語、抽象語、ネットスラング、皮肉、大人向けの笑いは避ける。
- ダジャレだけに頼らない。
- 人を傷つける、容姿をばかにする、差別的な表現は避ける。
- お題に人の弱点が含まれる場合も、その人を笑うのではなく特徴の見方を変えて笑えるようにする。
- why は子ども向けに1文で「なぜその言い換えになるか」が分かるようにする。
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
        f"なまえカエル。{answers['nickname']['answer']}。"
        f"ぎゃくてんカエル。{answers['twist']['answer']}。"
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
    "support_open": False,
    "support_request": "",
    "support_level": 0,
    "support_result": None,
    "support_audio": None,
    "support_autoplay_pending": False,
    "support_log": [],
    "learned_words": [],
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
        if key.startswith(("take_", "transcript_", "review_audio_")):
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


def voice_fill_text(field_key, audio_key, label, placeholder, context=""):
    current = str(st.session_state.get(field_key, ""))
    st.markdown(f"**{label}**")
    audio = st.audio_input(f"🎤 {label}を話してね", sample_rate=16000, key=audio_key)

    digest_key = f"_{audio_key}_digest"
    if audio is not None:
        digest = audio_digest(audio)
        if digest and st.session_state.get(digest_key) != digest:
            try:
                boosted = boost_recorded_wav(audio)
                audio_for_transcription = io.BytesIO(boosted.getvalue())
                audio_for_transcription.name = "recording_boosted.wav"
                with st.spinner(f"{label}を聞いています…"):
                    transcript = transcribe_audio(audio_for_transcription, context)
                if transcript:
                    st.session_state[field_key] = transcript
                    st.session_state[digest_key] = digest
                    current = transcript
                else:
                    st.warning(f"{label}をうまく聞き取れませんでした。もう一度話してください。")
            except Exception as exc:
                st.error(f"{label}の聞き取りに失敗しました。もう一度試してください。")
                with st.expander("保護者向け詳細"):
                    st.code(str(exc))

    value = st.text_input(
        label,
        value=str(st.session_state.get(field_key, current)),
        placeholder=placeholder,
        key=f"edit_{field_key}",
        label_visibility="collapsed",
    )
    st.session_state[field_key] = value
    return value


def apply_support_result(child_request, level, result):
    st.session_state.support_request = child_request
    st.session_state.support_level = level
    st.session_state.support_result = result
    st.session_state.support_audio = speech_bytes(result["message"])
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


def render_player_answers(answers):
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


# ============================================================
# Main UI
# ============================================================
verify_setup()
require_family_pin()

st.title("🐸 言いカエル おたすけAI")
st.caption("困ったときはヒント係。遊ぶときはAIもプレイヤー。")
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
    st.caption("お題と『言い方』は声で入れられます。聞き取ったあと、下の欄で直せます。")

    base_context = "カードゲーム『言いカエル』の入力です。短い言葉やフレーズとして自然に文字起こししてください。"
    topic = voice_fill_text(
        field_key=f"topic_draft_{st.session_state.round_serial}",
        audio_key=f"topic_audio_{st.session_state.round_serial}",
        label="お題カード",
        placeholder="例：走るのが遅い人",
        context=base_context + " 今はお題カードです。",
    )
    style = voice_fill_text(
        field_key=f"style_draft_{st.session_state.round_serial}",
        audio_key=f"style_audio_{st.session_state.round_serial}",
        label="言い方カード",
        placeholder="例：カッコよく",
        context=base_context + " 今は言い方カードです。",
    )
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


# -------------------- AI player mode --------------------
st.subheader("② AIもゲームに参加")
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
        render_player_answers(st.session_state.ai_answers)
        if not st.session_state.ai_audio:
            if st.button("🔊 AIの答えを聞く", use_container_width=True):
                try:
                    with st.spinner("声を作っています…"):
                        st.session_state.ai_audio = speech_bytes(
                            player_speech_text(st.session_state.ai_answers)
                        )
                    st.rerun()
                except Exception as exc:
                    st.error("読み上げ音声を作れませんでした。")
                    with st.expander("保護者向け詳細"):
                        st.code(str(exc))
        else:
            st.audio(st.session_state.ai_audio, format="audio/wav")


st.divider()
if st.session_state.learned_words:
    st.caption("このラウンドで出会ったことば：" + "・".join(st.session_state.learned_words))

if st.button("つぎのお題へ", type="primary", use_container_width=True):
    reset_round()
    st.rerun()
