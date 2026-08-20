# 言いカエル おたすけAI

家庭で「言いカエル」を遊ぶときの非公式サポートアプリです。

## できること

### 1. こまったときのサポート
子どもがマイクで、

- 「この言葉ってなに？」
- 「思いつかない」
- 「こんな感じにしたいけど言葉が出ない」

などと話します。

AIは最初から完成回答を出さず、4段階で助けます。

1. 見方を変えるヒント
2. 使えそうな言葉
3. 言葉の組み立て方
4. 最後だけ短い完成例を1つ

新しい言葉は、意味だけでなく「頭に浮かぶ場面」とセットで示します。

### 2. AIプレイヤー
AI自身もゲームに参加します。

毎回、次の3方向で回答します。

- たとえカエル：動物・乗り物・食べ物などにたとえる
- なまえカエル：あだ名・称号・キャラクター名にする
- ぎゃくてんカエル：弱点や特徴を別の見方に変える

AIの回答はラウンド開始時に先に作って伏せておき、人間が考えたあとに公開できます。
サポートAIにはAIプレイヤーの回答を渡さないため、ヒント側から答えが漏れにくい構成です。

## 前回の「にっき × ことばあそび」と同じ基本構成

Android Chrome
→ Streamlit Community Cloud
→ OpenAI API
→ Supabase（任意・履歴保存）

毎日の利用時にPCは不要です。

## ファイル

- `app.py`：本体
- `requirements.txt`：Python依存パッケージ
- `supabase_schema.sql`：履歴を保存する場合のテーブル
- `secrets.toml.example`：Streamlit Secretsの例
- `.streamlit/config.toml`：Streamlit設定

## 1. GitHub Repositoryを用意

新しいPrivate Repositoryを作り、このフォルダの中身をアップロードします。

本物の `.streamlit/secrets.toml` はGitHubへ置かないでください。

## 2. Streamlit Community Cloudへデプロイ

- Repository：上で作ったRepository
- Branch：`main`
- Main file path：`app.py`
- Python：3.12

Advanced settings > Secrets に `secrets.toml.example` を参考に値を登録します。

最低限必要なのは、

- `OPENAI_API_KEY`

です。

家族だけで使う場合は、

- `FAMILY_PIN`

も設定してください。

## 3. Supabase履歴を使う場合

以前の `kotoba_app` と同じSupabase Projectをそのまま再利用できます。

Supabase > SQL Editor > New query で `supabase_schema.sql` を実行します。
これは `iikaeru_rounds` テーブルだけを追加し、既存の `diaries` テーブルや `diary-images` Storageには触れません。

Streamlit Secretsに、

- `SUPABASE_URL`
- `SUPABASE_SECRET_KEY`

を追加すると「これまで」画面が自動的に有効になります。

Supabaseを設定しなくてもゲーム機能は動きます。

## 4. Androidで使う

Streamlit Community CloudのURLをAndroid Chromeで開きます。
初回だけマイク利用を許可します。

Chromeメニューから「ホーム画面に追加」すると、以後はアプリのように起動できます。

## 保存するもの

Supabaseを有効にした場合、以下を保存します。

- 日付
- お題
- 言い方
- サポートを使った内容
- サポートで出会った言葉
- AIプレイヤーの3回答
- AI回答を公開したか

保存しないもの：

- 子どもの録音音声
- AI読み上げ音声

## 使用モデル（初期値）

- 言葉の生成：`gpt-5.6-luna`
- 文字起こし：`gpt-4o-mini-transcribe`
- 読み上げ：`gpt-4o-mini-tts`
- 声：`coral`

モデル名はStreamlit Secretsから変更できます。

## 最初の確認

本番利用前に保護者が1回、

1. お題・言い方を入力
2. サポートを音声で呼ぶ
3. 文字起こしを確認して送る
4. ヒントが読み上げられる
5. 「もう少しヒント」が4段階で進む
6. AIの回答を公開する
7. AIの3回答が別方向になっている
8. 次のお題へ進める

まで通してください。
