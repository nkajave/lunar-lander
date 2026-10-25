# Lunar Lander: land it, then teach an AI

**Live site:** https://nkajave.github.io/lunar-lander/

An interactive reinforcement-learning playground that runs entirely in the browser:

- **Fly it yourself** with the keyboard or on-screen buttons.
- **Watch a trained AI**: three PPO agents trained in Python, each rewarded differently. One of them learned to cheat.
- **Train your own AI** live in your browser. Pick what to reward, press start, and watch it go from crashing to landing (or to hovering forever, if you reward the wrong thing).

## How the website works

| File | What it does |
|---|---|
| `index.html`, `css/style.css` | The page |
| `js/lander.js` | Gymnasium's LunarLander-v3 ported to JavaScript with [planck.js](https://piqnt.com/planck.js/) (Box2D) |
| `js/nn.js` | A tiny neural network and Adam optimizer, no libraries |
| `js/ppo.js` | PPO, matching Stable-Baselines3's settings |
| `js/train-worker.js` | Runs training in a background thread so the page stays smooth |
| `js/render.js`, `js/sound.js`, `js/chart.js`, `js/app.js` | Drawing, sound effects, learning curve, controls |
| `models/*.json` | The three Python-trained policies, exported for the browser |

The port is faithful: the Python-trained models score about the same in the browser (default 230, hover 72, fixed 264) as in Python (235, 66, 260).

## The Python experiment

| File | What it does |
|---|---|
| `train.py` | Train PPO with a chosen reward: `default`, `time`, `hover`, `fixed` |
| `rewards.py` | The reward wrappers |
| `evaluate.py` | Score a model on the plain game, watch it, or record it |
| `record_with_sound.py` | Record clips with synthesized engine sounds |
| `watch_random.py` | Watch random actions crash |

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install swig
pip install -r requirements.txt
python train.py --reward hover
python evaluate.py ppo_hover_seed0 --watch
```

### Results (1M frames, 20 test attempts, bonus switched off)

| Rewarded for | Avg. score | Avg. frames | Landed |
|---|---|---|---|
| The game's own score | 235 | 464 | 20/20 |
| +1 for every steady frame | 66 | 950 | 3/20 |
| Paid for getting steady | 260 | 346 | 20/20 |

Paying +1 for every steady frame lets a hovering rocket farm up to 1,000 points, while landing ends the attempt and stops the payments. So the AI learns to hover. Paying for *becoming* steady caps the bonus at 1 per attempt and fixes it.

## Run the website locally

Browsers block background workers on `file://` pages, so serve the folder:

```bash
python -m http.server
```

Then open http://localhost:8000.
