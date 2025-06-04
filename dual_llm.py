import openai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import json
import os
from dotenv import load_dotenv

# 環境変数からAPIキーを読み込む
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 裏メモリー用の埋め込みモデルと記憶
embedder = SentenceTransformer("all-MiniLM-L6-v2")
裏メモリー = []
メモリーファイル = "ura_memory.json"

# JSON保存・読み込み
def メモリー保存():
    with open(メモリーファイル, "w", encoding="utf-8") as f:
        json.dump([
            {
                "user": u,
                "表": s,
                "裏": d,
                "vec": v.tolist()
            } for u, s, d, v in 裏メモリー
        ], f, ensure_ascii=False, indent=2)

def メモリー読み込み():
    if os.path.exists(メモリーファイル):
        with open(メモリーファイル, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                裏メモリー.append((
                    item["user"],
                    item["表"],
                    item["裏"],
                    torch.tensor(item["vec"])
                ))

# GPT-3.5での応答生成
def generate_openai(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "あなたは親しみやすく自然なAIです。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=200,
    )
    return response["choices"][0]["message"]["content"]

# 記録保存
def 記録する(ユーザー入力, 表出力, 裏出力):
    embedding = embedder.encode(ユーザー入力 + " " + 表出力 + " " + 裏出力)
    裏メモリー.append((ユーザー入力, 表出力, 裏出力, torch.tensor(embedding)))
    メモリー保存()

# 思い出す機能
def 思い出す(新しい入力):
    if not 裏メモリー:
        return None
    query_vec = embedder.encode(新しい入力)
    sims = [cosine_similarity([query_vec], [v[3]])[0][0] for v in 裏メモリー]
    top_index = sims.index(max(sims))
    return 裏メモリー[top_index] if sims[top_index] > 0.6 else None

# メイン関数（テスト用）
def 対話する():
    メモリー読み込み()
    while True:
        user_input = input("ユーザー: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # 表の応答
        surface_prompt = f"""以下はユーザーとAIの会話です。
ユーザー: {user_input}
AI:"""

        思い出 = 思い出す(user_input)
        におわせ = ""
        if 思い出:
            におわせ = f"（以前も似た話があったような気がします…）"

        表出力 = generate_openai(surface_prompt + "\n" + におわせ)

        # 裏の思考
        deep_prompt = f"""ユーザーの発言: {user_input}
表の返答: {表出力}
この会話の裏でAIが考えていることを、感情・観察・意図の視点で記述してください："""
        裏出力 = generate_openai(deep_prompt)

        記録する(user_input, 表出力, 裏出力)

        print("【表】:", 表出力.strip())
        print("【裏】:", 裏出力.strip())

# スタート
if __name__ == "__main__":
    対話する()
