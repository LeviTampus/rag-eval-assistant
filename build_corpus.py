from datasets import load_dataset
from pathlib import Path

def build_corpus(n: int = 40, out_dir: str = "data/docs"):
    ds = load_dataset("TheRealVigilante/Processed_Top_15k_Anime",split="train")
    rows = []
    for r in ds:
        if r["rank"] > 0:
            rows.append(r)

    rows = sorted(rows, key=lambda r: r["rank"])
    top = rows[:n]
    for anime in top:
        content = "\n".join([
            f"Title: {anime['name']}",
            f"Genres: {' '.join(anime['genres'].split())}",
            f"Studios: {' '.join(anime['studios'].split())}",
            f"Year: {' '.join((anime['premiered'] or 'Unknown').split())}",
            f"Synopsis: {' '.join(anime['synopsis'].split())}",
        ])
        Path(f"{out_dir}/{anime['name']}.txt").write_text(content)

    print(f"Wrote {len(top)} files to {out_dir}/")


if __name__ == "__main__":
    build_corpus()




