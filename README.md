# Towards High-Quality One-Shot Handwriting Generation with Patch Contrastive Enhancement and Style-Aware Quantization

## Abstract
Styled handwriting image generation, despite achieving impressive results in recent years, remains challenging due to the difficulty in capturing the intricate and diverse characteristics of human handwriting. Additionally, the requirement to generate using solely a single reference image in practical settings further exacerbates the complexity of the problem. Existing methods still struggle to generate visually appealing and realistic handwritten images and adapt to complex unseen writer styles, struggling to isolate invariant style features (e.g., slant, stroke width, curvature) while ignoring irrelevant noise. To tackle this problem, we introduce Patch **Con**trastive Enhancement and **St**yle-**A**ware Qua**nt**ization via Denoising Diffusion (**CONSTANT**), a novel one-shot handwriting generation via diffusion model. CONSTANT leverages three key innovations: 1) a Style-Aware Quantization (SAQ) module that models style as discrete visual tokens capturing distinct concepts; 2) a contrastive objective to ensure these tokens are well-separated and meaningful in the embedding style space; 3) a latent patch-based contrastive (
) objective help improving quality and local structures by aligning multiscale spatial patches of generated and real features in latent space. Extensive experiments and analysis on benchmark datasets from multiple languages, including English and Chinese, and our proposed ViHTGen dataset for Vietnamese, demonstrate the superiority of adapting to new reference styles and producing highly detailed images of our method over state-of-the-art approaches. Code is available at [https://github.com/anonymous6399/CONSTANT](https://github.com/anonymous6399/CONSTANT)

![mainflow](assets/mainflow.png)

## Training

```bash
pip install -r requirements.txt
```

```bash
python train.py --config-path config/constant.yaml
```

