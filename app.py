import streamlit as st
from pathlib import Path
from PIL import Image
import re

st.set_page_config(page_title="EEG-to-Image Reconstruction", layout="wide")

BASE_DIR = Path(__file__).parent
TOP50_1000_DIR = BASE_DIR / "top50_1000"
TOP50_BEST_DIR = BASE_DIR / "top50_best"
RECON_COMBINED_DIR = BASE_DIR / "reconstructions_combined"


@st.cache_data
def load_top50_best_old():
    results = {}
    if TOP50_BEST_DIR.exists():
        for f in TOP50_BEST_DIR.glob("rank*_clip*.png"):
            if "_GT" in f.name:
                continue
            match = re.search(r'rank\d+_(\d+)_.*_clip([\d.]+)\.png', f.name)
            if match:
                idx = int(match.group(1))
                score = float(match.group(2))
                if idx not in results or score > results[idx]['score']:
                    results[idx] = {'path': f, 'score': score}
    return results


@st.cache_data
def load_data():
    samples = {}

    if TOP50_1000_DIR.exists():
        for f in TOP50_1000_DIR.glob("idx*_GT.png"):
            match = re.search(r'idx(\d+)_(.+)_GT\.png', f.name)
            if match:
                idx = int(match.group(1))
                category = match.group(2)
                samples[idx] = {'category': category, 'gt': f, 'recons': []}

        for f in TOP50_1000_DIR.glob("idx*_top*_clip*.png"):
            match = re.search(r'idx(\d+)_.*_top(\d+)_clip([\d.]+)_(.+)\.png', f.name)
            if match:
                idx = int(match.group(1))
                rank = int(match.group(2))
                score = float(match.group(3))
                subject = match.group(4)
                if idx in samples:
                    samples[idx]['recons'].append({'rank': rank, 'score': score, 'subject': subject, 'path': f})

        for idx in samples:
            samples[idx]['recons'] = sorted(samples[idx]['recons'], key=lambda x: x['rank'])

    top50_best_old = load_top50_best_old()

    for idx in samples:
        if idx in top50_best_old:
            samples[idx]['old_200'] = top50_best_old[idx]
        else:
            samples[idx]['old_200'] = None

        combined_path = RECON_COMBINED_DIR / f"reconstruction_{idx:03d}.png"
        if combined_path.exists():
            samples[idx]['old_50'] = {'path': combined_path}
        else:
            samples[idx]['old_50'] = None

    return samples


def main():
    st.title("🧠 EEG-to-Image Reconstruction")
    st.markdown("**Top 50** | New (1000 attempts): 0.661 | Old (200): 0.627 | Original (50): 0.482")

    samples = load_data()

    if not samples:
        st.error("No samples found")
        return

    sorted_indices = sorted(
        samples.keys(),
        key=lambda x: samples[x]['recons'][0]['score'] if samples[x]['recons'] else 0,
        reverse=True
    )

    n_show = st.sidebar.slider("Show samples", 5, 50, 20)

    for idx in sorted_indices[:n_show]:
        sample = samples[idx]
        best_score = sample['recons'][0]['score'] if sample['recons'] else 0

        st.subheader(f"#{idx} {sample['category']} (best: {best_score:.3f})")

        row1 = st.columns(4)

        with row1[0]:
            st.image(Image.open(sample['gt']), caption="Ground Truth", use_container_width=True)

        for i in range(3):
            with row1[i + 1]:
                if i < len(sample['recons']):
                    recon = sample['recons'][i]
                    st.image(
                        Image.open(recon['path']),
                        caption=f"New #{recon['rank']} | {recon['score']:.3f}",
                        use_container_width=True
                    )

        row2 = st.columns(4)

        for i in range(2):
            with row2[i]:
                if i + 3 < len(sample['recons']):
                    recon = sample['recons'][i + 3]
                    st.image(
                        Image.open(recon['path']),
                        caption=f"New #{recon['rank']} | {recon['score']:.3f}",
                        use_container_width=True
                    )

        with row2[2]:
            if sample['old_200']:
                st.image(
                    Image.open(sample['old_200']['path']),
                    caption=f"Old 200 | {sample['old_200']['score']:.3f}",
                    use_container_width=True
                )

        with row2[3]:
            if sample['old_50']:
                st.image(
                    Image.open(sample['old_50']['path']),
                    caption="Original 50",
                    use_container_width=True
                )

        st.divider()


if __name__ == "__main__":
    main()
