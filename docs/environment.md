# The headline gap: running the notebooks

Use Python 3.11 for a straightforward setup. The notebook check prints the executable used by the kernel, installed package versions and any missing requirements. Run that check before the analysis cells.

## What each notebook needs

| Notebook | Requirements file | Training device |
|---|---|---|
| 01: EDA and classical models | `requirements-ml.txt` | CPU |
| 02: DistilBERT | `requirements.txt`, which includes the first file | CPU or CUDA GPU |

The project uses Transformers `>=4.57,<6`, Accelerate `>=1.1,<2` and PyTorch `>=2.6,<3`. The broad major-version limits allow a current Colab installation and the local environment to use the same notebook API. They are not a frozen environment. The locally checked combination is Python 3.11.13, PyTorch 2.11.0+cpu, Transformers 5.7.0 and Accelerate 1.15.0. A short training, checkpoint save/reload and inference check passed using a small randomly initialized DistilBERT and synthetic inputs. This validates execution rather than project benchmark quality. The completed Colab evaluation used a Tesla T4, Python 3.13.16, PyTorch 2.11.0+cu130, Transformers 5.21.0 and Accelerate 1.15.0. These are the versions printed in the saved run.

## Local CPU setup

From the repository root, create an environment with Python 3.11. For Conda:

```bash
conda create -n fake-news python=3.11
conda activate fake-news
python -m pip install -r requirements-ml.txt
python -m ipykernel install --user --name fake-news --display-name "Python (fake-news)"
```

Select **Python (fake-news)** as the notebook kernel in your editor. Notebook 1 can now run independently of the deep learning libraries.

For notebook 2, install the CPU build of PyTorch first, then the complete requirements:

```bash
python -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Restart the notebook kernel after installing these libraries. Run the environment check again, then continue from the first configuration cell. The CPU wheel command follows the [official Transformers CPU installation instructions](https://huggingface.co/docs/transformers/v4.57.1/installation); [PyTorch's installer](https://pytorch.org/get-started/locally/) covers other local hardware.

If you already use an environment, select its kernel and install from a notebook cell so the installation targets that Python:

```python
# When the notebook working directory is notebooks/:
%pip install --upgrade -r ../requirements.txt
```

Use `requirements.txt` without `../` when the working directory is the repository root. Restart the kernel after this command. If the environment check passes in a fresh session, continue with the analysis.

## Colab GPU setup

1. Copy the complete repository folder to Google Drive, including `data/evaluation/`. To compare with the completed classical run, include `results/metrics/01_classical_ml/`.
2. Open notebook 2 and choose a GPU under the runtime hardware settings.
3. Mount Drive and install dependencies before running the environment check:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   %pip install -r /content/drive/MyDrive/fake-news-project/requirements.txt
   ```

4. Restart the runtime after installation. Run the notebook from the top. Set `USE_COLAB=True` and adjust `PROJECT_ROOT` to the folder copied to Drive.
5. Keep `DEVICE_PREFERENCE='auto'`. The device cell should print `cuda` and the GPU name. If it prints `cpu`, check the runtime hardware setting. Setting `DEVICE_PREFERENCE='cuda'` requests an explicit error when no CUDA device is available.

Keep Colab's CUDA-enabled PyTorch if it satisfies the requirements. The local CPU wheel installation command is intended for the local CPU environment. Notebook 1 runs on CPU in Colab as well; it does not benefit from selecting a GPU.

## CPU and GPU behavior in notebook 2

| Setting | Local CPU | Colab CUDA GPU |
|---|---:|---:|
| Per-device training batch | 4 | 16 |
| Gradient accumulation | 4 | 1 |
| Effective training batch | 16 | 16 |
| Evaluation batch | 8 | 16 |
| Worker subprocesses | 0 | 0 |
| Precision | float32 | float32 |

Both modes use the complete training partition and the same maximum epoch count. CPU training is supported but can take considerably longer. The practical workflow is GPU fine-tuning and CPU evaluation of the saved model. Hardware and library versions can still change numerical results.

The [Trainer documentation](https://huggingface.co/docs/transformers/v5.7.0/en/main_classes/trainer) describes its device and batching settings. The notebook sets `use_cpu` explicitly from the selected device and reserves validation for checkpoint selection.

## Move a trained model from Colab to the local project

After training, copy this entire directory from Drive to the same relative location locally:

```text
distilbert_model/split_60_15_25_v1/
```

Include model weights, configuration, tokenizer files and `evaluation_protocol.json`. Keep the same input CSV files and `FORCE_RETRAIN=False`. On the local CPU, the training cell detects and reuses the saved model. A checkpoint with different recorded data or partitions is rejected.

The custom-set comparison also reads `results/metrics/01_classical_ml/`. If these files are missing or the partitions differ, notebook 2 displays its own result and leaves the cross-model conclusion pending.

## Recognize the messages

| Message | Meaning and action |
|---|---|
| `Trainer ... requires accelerate>=1.1.0` | Install a supported Accelerate version in the current kernel, restart it and run from the top. If Accelerate was installed during the session, Transformers may still have cached its earlier availability. |
| Environment check finds a different Python or missing package | Select the intended kernel. `%pip` targets the kernel currently running the cell. |
| No `HF_TOKEN` configured | Public DistilBERT downloads can work without a token. Authentication can raise download limits; a rate-limit or access error needs separate attention. |
| Unused pretraining weights / newly initialized classifier weights | Expected when initializing a classification head from the base pretrained model. The new head needs fine-tuning. |
| `cuda` unavailable | Use CPU or select a Colab GPU runtime. Installing Accelerate does not turn a CPU-only PyTorch installation into a CUDA installation. |
| Custom loader finds two examples instead of 40 | Check the printed `CUSTOM_PATH`. Replace `data/evaluation/custom_headlines.json` under that project folder on Drive with the repository copy. Rerun custom loading, inference and result cells. Keep the trained checkpoint. |
| `Creating 40 fallback records` or custom fingerprint mismatch | The original evaluation set has been replaced with substitute data. Restore the repository JSON at the printed path. Counts and IDs alone do not identify a benchmark; the loader also verifies texts and labels. Rerun the custom section with the saved model. |

## Indicazioni in italiano

Nel primo notebook le dipendenze sono quelle di `requirements-ml.txt`. Il secondo aggiunge PyTorch, Transformers, Datasets e Accelerate attraverso `requirements.txt`.

Il primo controllo stampa il percorso di Python: deve corrispondere all'ambiente selezionato nell'editor. Dopo un'installazione, riavviare il kernel e ripartire dall'inizio. Questo passaggio conta anche quando `pip` dice che Accelerate è già presente: una sessione aperta può conservare il vecchio controllo delle dipendenze.

In locale, `DEVICE_PREFERENCE='auto'` sceglie la CPU quando CUDA è assente. Il training è consentito, ma può essere lento. Su Colab selezionare una GPU, installare le dipendenze dalla cartella del progetto su Drive, riavviare il runtime e impostare `USE_COLAB=True`. La cella del dispositivo deve mostrare `cuda`.

Per addestrare su Colab e valutare in locale, copiare l'intera cartella del modello indicata sopra, conservare gli stessi CSV e lasciare `FORCE_RETRAIN=False`. Per il confronto finale servono anche le metriche del primo notebook. Le domande di ricerca hanno lo stesso testo nei due notebook: il primo risponde alle prime due e prepara la baseline; il secondo completa la terza con i risultati misurati.
