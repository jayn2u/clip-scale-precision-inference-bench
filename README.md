# CLIP Scale & Precision Inference Benchmark

OpenAI CLIP 모델의 **크기(scale)**와 **가중치 정밀도(weight precision)**가 zero-shot 텍스트→이미지 사람 검색에 미치는 영향을 비교합니다. CUHK-PEDES 테스트 split을 사용하며, 검색 정확도뿐 아니라 처리량, GPU 메모리, fp32 기준 수치 오차와 순위 변화를 함께 측정합니다.

## 벤치마크 범위

| 구분 | 측정 대상 |
| --- | --- |
| 모델 | `ViT-B/32`, `ViT-B/16`, `ViT-L/14`, `RN50` |
| 정밀도 | `fp32`, `fp16`, `bf16` |
| 검색 품질 | R@1, R@5, R@10, mAP |
| 성능 및 메모리 | 이미지/텍스트 처리량, 전체 소요 시간, peak VRAM, 가중치 크기 |
| fp32 대비 변화 | 임베딩 cosine similarity, Top-1 일치율, Top-10 overlap, Kendall tau |

이 프로젝트는 zero-shot 추론 비교만 다룹니다. CUHK-PEDES fine-tuning, INT8/FP8 양자화, TensorRT 최적화는 현재 범위에 포함하지 않습니다.

## 요구사항

- Python 3.12 이상
- [`uv`](https://docs.astral.sh/uv/)
- CUDA를 지원하는 NVIDIA GPU
- CUDA 12.8 호환 드라이버
- CUHK-PEDES 데이터셋

PyTorch와 torchvision은 `pyproject.toml`에 지정된 CUDA 12.8 wheel로 설치됩니다. 현재 구현은 타이밍과 메모리 측정에 `torch.cuda`를 직접 사용하므로 실질적으로 CUDA 환경이 필요합니다. `--device cpu`는 완전한 CPU 실행 모드가 아닙니다.

## 빠른 시작

### 1. 의존성 설치

```bash
uv sync
```

OpenAI CLIP 체크포인트는 최초 실행 시 자동으로 내려받습니다. 기본 캐시는 `~/.cache/clip`이며 `CLIP_DOWNLOAD_ROOT`로 변경할 수 있습니다.

### 2. 데이터셋 준비

데이터 경로는 `/mnt/data/lab_datasets/CUHK-PEDES`와 `/data/jayn2u/lab_datasets/CUHK-PEDES`에서 자동 탐색합니다. 다른 위치에 두었다면 `CUHK_PEDES_ROOT`를 지정합니다.

```text
CUHK-PEDES/
├── reid_raw.json
└── imgs/
    └── ...
```

```bash
export CUHK_PEDES_ROOT=/path/to/CUHK-PEDES
```

`reid_raw.json` 안의 `file_path` 값은 `imgs/`를 기준으로 해석됩니다. 표준 test split은 이미지 3,074장, 캡션 6,156개, 인물 ID 1,000개로 구성됩니다.

### 3. 스모크 테스트

먼저 작은 subset으로 설치와 데이터 경로를 확인합니다.

```bash
uv run clipbench run --limit 200
```

`--limit 200` 결과는 자동으로 `smoke200` namespace에 저장되므로 전체 실험 캐시와 섞이지 않습니다. 단, 작은 실행의 처리량은 partial batch를 사용할 수 있어 정식 성능 비교 자료로 사용하면 안 됩니다.

### 4. 전체 벤치마크

```bash
uv run clipbench run
```

실행 순서는 `encode` → `eval` → `report`이며, 결과 보고서는 `results/report/report_full.html`에 생성됩니다.

## 하드웨어와 결과 비교

저장소에 커밋된 결과는 **RTX A6000(48 GiB)**에서 측정했습니다. 각 metrics JSON 행에 `cuda_device`와 `torch_version`이 기록됩니다. 검색 정확도는 두 GPU의 fp32 결과가 소수 둘째 자리까지 재현됐지만, 처리량과 precision에 따른 속도 향상 비율은 하드웨어에 종속됩니다. 예를 들어 ViT-L/14 fp16의 fp32 대비 속도는 RTX 5070 Ti에서 2.80배, A6000에서 3.81배였습니다. 서로 다른 GPU에서 얻은 속도 열을 한 실험처럼 섞어 비교하지 마세요.

## 명령어

```bash
uv run clipbench encode [옵션]  # 임베딩 생성 및 캐시
uv run clipbench eval [옵션]    # 검색 및 drift 지표 계산
uv run clipbench report [옵션]  # 기존 지표로 HTML 보고서 생성
uv run clipbench run [옵션]     # 위 세 단계를 순서대로 실행
```

모든 subcommand에서 다음 옵션을 사용할 수 있습니다.

| 옵션 | 설명 |
| --- | --- |
| `--models MODEL ...` | 실행할 CLIP 모델 목록 |
| `--dtypes {fp32,fp16,bf16} ...` | 실행할 정밀도 목록 |
| `--limit N` | test 이미지 앞 N장만 사용하는 스모크 실행 |
| `--tag NAME` | 캐시와 결과 namespace 직접 지정 |
| `--force` | 유효한 캐시가 있어도 다시 인코딩 |
| `--device DEVICE` | 모델을 올릴 장치. 기본값은 `cuda` |
| `--quiet` | 진행률 표시 비활성화 |

예시:

```bash
# 한 모델의 fp32/fp16만 비교
uv run clipbench run --models ViT-B/32 --dtypes fp32 fp16 --tag vit-b32

# 임베딩을 재사용하여 평가와 보고서만 다시 생성
uv run clipbench eval --tag full
uv run clipbench report --tag full

# 기존 캐시를 무시하고 다시 인코딩
uv run clipbench encode --models RN50 --dtypes fp32 --force
```

`eval`에서 fp32 대비 drift 지표를 계산하려면 같은 모델과 tag의 fp32 임베딩이 반드시 있어야 합니다. 비교 실험에서는 `fp32`를 `--dtypes` 목록에 포함하세요.

## 출력 구조와 캐시

```text
results/
├── embeddings/<tag>/<model>/<dtype>/
│   ├── image.npy
│   ├── text.npy
│   └── meta.json
├── metrics/results_<tag>.json
└── report/report_<tag>.html
```

- 임베딩은 측정한 모델의 정밀도와 관계없이 항상 fp32 NumPy 배열로 저장됩니다. 저장 단계에서 다시 양자화해 drift 측정을 오염시키지 않기 위해서입니다.
- 캐시에는 데이터 항목의 정체성과 순서를 포함한 hash가 기록됩니다. 현재 split과 일치하지 않는 캐시는 자동으로 무효 처리됩니다.
- `--tag`가 없으면 전체 실행은 `full`, `--limit N` 실행은 `smokeN`을 사용합니다.
- 용량이 큰 `results/embeddings/`, 로그, smoke/sanity 산출물은 Git에서 제외합니다.
- 재현 가능한 정식 실험의 metrics JSON과 HTML report는 프로젝트 결과물로 커밋할 수 있습니다.

## 측정 프로토콜

각 캡션을 query로 삼아 전체 test 이미지를 검색합니다. 정답 여부는 캡션이 붙은 원본 이미지 하나가 아니라 **person ID 일치**로 판단합니다. 같은 인물의 다른 이미지도 정답이며, 이는 CUHK-PEDES 논문 및 후속 person ReID 연구의 방식입니다.

- 이미지 전처리는 각 CLIP 모델의 기본 transform을 그대로 사용합니다.
- CLIP의 77-token context를 넘는 캡션은 제거하지 않고 truncate하며 개수를 기록합니다.
- 처리량은 warmup 이후 CUDA 동기화된 forward pass만 측정합니다.
- 마지막 partial batch를 제외한 full batch의 항목당 시간 중앙값을 우선 사용합니다.
- precision 사이의 비교가 batch 크기 차이에 오염되지 않도록 이미지 64, 텍스트 256의 고정 batch size를 사용합니다.

## 정밀도 구현에서 지키는 원칙

이 벤치마크는 `torch.autocast`가 아닌 **실제 가중치 casting**을 비교합니다. autocast는 fp32 master weight와 일부 fp32 연산을 유지하므로 이 프로젝트가 보려는 메모리 절감과 수치 drift를 정확히 나타내지 못합니다.

또한 `clip.load()`가 CUDA에서 fp16 가중치를 반환할 수 있으므로, 모든 모델은 먼저 명시적으로 fp32로 복원한 다음 목표 dtype으로 변환합니다. fp16과 bf16은 동일한 레이어 선택 정책을 사용하고 LayerNorm과 logit scale은 fp32로 유지합니다. 따라서 둘의 차이는 변환 레이어가 아니라 숫자 형식에서 비롯됩니다.

이 로직을 변경할 때는 다음 불변조건을 유지해야 합니다.

1. 모든 dtype은 동일한 fp32 기준 모델에서 시작해야 합니다.
2. fp16과 bf16은 동일한 종류의 레이어를 변환해야 합니다.
3. dtype별 batch size, 전처리, 데이터 순서, truncation 정책이 같아야 합니다.
4. 캐시된 임베딩은 fp32로 저장되어야 합니다.
5. fp32 결과가 없는 비교를 drift 결과로 해석하면 안 됩니다.

## 결과 해석

- R@K는 상위 K개 결과 안에 같은 person ID가 하나라도 있는 query 비율입니다.
- mAP는 같은 ID의 여러 정답 이미지가 전체 순위 어디에 놓였는지 반영합니다.
- embedding cosine은 벡터가 fp32 기준에서 얼마나 이동했는지 보여줍니다.
- Top-1 agreement와 Top-10 overlap은 그 이동이 실제 상위 검색 결과를 바꿨는지 보여줍니다.
- Kendall tau는 전체 순위 변화이며 계산 비용을 줄이기 위해 고정 seed로 선택한 최대 500개 query에서 측정합니다.

절대 검색 정확도는 fine-tuned 최신 모델과 직접 비교하지 마세요. 이 프로젝트의 주요 관심사는 한 모델 안에서 precision만 바꿨을 때 나타나는 상대적 차이입니다.

## 프로젝트 구조

```text
src/clipbench/
├── cli.py      # CLI와 encode/eval/report 실행 흐름
├── config.py   # 모델, dtype, batch size, 경로, 평가 설정
├── data.py     # CUHK-PEDES test split 로딩과 order hash
├── models.py   # 모델 로딩과 명시적 weight casting
├── encode.py   # 임베딩 생성, 성능 측정, 캐시
├── metrics.py  # 검색 품질과 fp32 대비 drift 계산
└── report.py   # 독립 실행 가능한 HTML 보고서 생성
```

설정을 바꿀 때는 우선 `src/clipbench/config.py`를 확인하세요. 새로운 모델 또는 정밀도를 추가하면 모델 로딩, 입력 dtype, 보고서 색상/표현까지 함께 검토해야 합니다.

## 변경 작업 가이드

1. 변경 전 작은 `--limit` 실행으로 재현 가능한 기준 결과를 확보합니다.
2. 구현 후 같은 모델·dtype·limit·tag 조건으로 다시 실행합니다.
3. 검색 지표뿐 아니라 parameter dtype histogram, weight bytes, peak VRAM, fp32 agreement도 함께 확인합니다.
4. protocol 또는 기본 설정을 바꾸면 기존 결과와 같은 실험인 것처럼 섞지 말고 새로운 `--tag`를 사용합니다.
5. 대용량 임베딩은 커밋하지 않습니다. 공유할 정식 실행의 metrics와 report는 코드·설정과 함께 검토합니다.

CLI 자체를 빠르게 확인하려면 GPU나 데이터셋 없이 다음 명령을 사용할 수 있습니다.

```bash
uv run clipbench --help
uv run clipbench run --help
```

현재 별도의 자동화 테스트 suite는 없습니다. 변경 사항은 최소한 CLI help 확인과 작은 스모크 benchmark로 검증하고, 계산 로직을 확장할 때는 단위 테스트 추가를 우선 고려하세요.
