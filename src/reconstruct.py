import os, csv, time
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from data.prompt import Reverse_prompt

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("LLM_API_KEY"),
    base_url=os.environ.get("LLM_API_BASE_URL"),
)


def reconstruct(output_text: str, temperature: float = 0.0) -> str:
    prompt = Reverse_prompt.format(output=output_text)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens= 300
    )
    return response.choices[0].message.content.strip()

def run_reconstruction(gold_path: str, out_path: str, temperature: float):

    with open(gold_path, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    fieldnames = ["id", "topic", "style", "level", "prompt", "output", "reconstructed_prompt", "temperature"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rec in tqdm(reader, desc=f"Reconstructing (t={temperature})"):

            try:
                recon = reconstruct(rec["output"], temperature)
            except Exception as e:
                print(f"Error on ID {rec['id']}: {e}")
                recon = "ERROR"

            writer.writerow({
                "id": rec["id"],
                "topic": rec["topic"],
                "style": rec["style"],
                "level": rec["level"],
                "prompt": rec["prompt"],
                "output": rec["output"],
                "reconstructed_prompt": recon,
                "temperature": temperature
            })

            time.sleep(2.5) 


if __name__ == "__main__":
    run_reconstruction(
        "data/gold_dataset.csv",
        "data/reconstructions_t0.7.csv",
        temperature=0.7
    )
    print("Reconstruction complete.")