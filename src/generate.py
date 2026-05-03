import os, csv, json, time
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from data.prompt import Answer_prompt

load_dotenv()
client = OpenAI(
    api_key=os.environ.get("LLM_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

def build_prompt(row):
    return Answer_prompt.format(
        Style=row["Style"],
        Level=row["Level"],
        Prompt=row["Prompt"]
    )

def generate_output(prompt: str, temperature: float = 0.7) -> str:
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error: {e}")
        return ""

def build_gold_dataset(prompts_csv: str, output_path: str):

    with open(prompts_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    fieldnames = ["id", "topic", "style", "level", "prompt", "output"]

    with open(output_path, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()

        for row in tqdm(rows, desc="Generating outputs"):

            # Build structured prompt
            full_prompt = build_prompt(row)

            # Generate response
            output = generate_output(full_prompt)

            # Write row
            writer.writerow({
                "id": row["ID"],
                "topic": row["Topic"],
                "style": row["Style"],
                "level": row["Level"],
                "prompt": row["Prompt"],
                "output": output
            })

            time.sleep(0.6)
            
if __name__ == "__main__":
    build_gold_dataset("data/prompts.csv", "data/gold_dataset.csv")
    print("Done. Gold dataset saved.")