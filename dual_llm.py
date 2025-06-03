from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import json
import os

# モデルとトークナイザー
surface_model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(surface_model_name)
model_surface = AutoModelForCausalLM.from_pretrained(surface_model_name)
model_deep = AutoModelForCausalLM.from_pretrained(surface_model_name)

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

# 文章生成関数
def generate_response(model, prompt, tokenizer, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            input_ids, max_length=max_length,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

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
        surface_prompt = f"ユーザー: {user_input}\nAI: "
        表出力 = generate_response(model_surface, surface_prompt, tokenizer)

        # 裏の思考
        deep_prompt = f"""ユーザーの発言: {user_input}
表の返答: {表出力}
この会話の裏でAIが考えていることを、感情・観察・意図の視点で記述してください："""
        裏出力 = generate_response(model_deep, deep_prompt, tokenizer)

        記録する(user_input, 表出力, 裏出力)

        print("【表】:", 表出力.strip())
        print("【裏】:", 裏出力.strip())

# スタート
if __name__ == "__main__":
    対話する()
