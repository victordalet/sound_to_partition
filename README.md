# sound_to_partition

---

## Train the model

---

```bash 
export PYTHONPATH=$(pwd)/src:$(pwd)
python src/train.py
```

## Use

---

```bash
export PYTHONPATH=$(pwd)/src:$(pwd)
python src/run.py {audio.mp3} {result.csv}
```