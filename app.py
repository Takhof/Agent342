from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

# モデルとトークナイザー
surface_model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(surface_model_name)
model_surface = AutoModelForCausalLM.from_pretrained(surface_model_name)
model_deep = AutoModelForCausalLM.from_pretrained(surface_model_name)

# 裏メモリー用の埋め込みモデルと記憶
embedder = SentenceTransformer("all-MiniLM-L6-v2")
裏メモリー = []

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
    裏メモリー.append((ユーザー入力, 表出力, 裏出力, embedding))

# 思い出す機能
def 思い出す(新しい入力):
    if not 裏メモリー:
        return None
    query_vec = embedder.encode(new_input)
    sims = [cosine_similarity([query_vec], [v[-1]])[0][0] for v in 裏メモリー]
    top_index = sims.index(max(sims))
    return 裏メモリー[top_index] if sims[top_index] > 0.6 else None

# メイン関数（テスト用）
def 対話する():
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
