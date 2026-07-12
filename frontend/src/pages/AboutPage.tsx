import { Card } from '../components/ui/Card';

const PIPELINE_STAGES = [
  {
    name: 'Preprocessing',
    text: 'Validation, adaptive bilateral denoising, and selective gamma lift on very dark regions.',
  },
  {
    name: 'Glow Detection',
    text: 'Locates artificial light sources and their halos, and estimates each one’s local atmospheric light.',
  },
  {
    name: 'FFA-Net Inference',
    text: 'Pretrained feature-fusion attention network produces the deep-learning dehazed estimate.',
  },
  {
    name: 'Radiance Recovery',
    text: 'Inverts the atmospheric scattering model using a per-region atmospheric light map.',
  },
  {
    name: 'Post-processing',
    text: 'CLAHE local contrast, light denoising, unsharp masking, and a saturation boost.',
  },
  {
    name: 'Quality Assessment',
    text: 'Computes full-reference and no-reference metrics comparing input to output.',
  },
];

const METRICS = [
  {
    name: 'PSNR',
    text: 'Peak Signal-to-Noise Ratio — pixel-level fidelity between the input and the result. Higher is better.',
  },
  {
    name: 'SSIM',
    text: 'Structural Similarity — how well perceptual structure is preserved. Higher is better.',
  },
  {
    name: 'NIQE',
    text: 'Naturalness Image Quality Evaluator — a no-reference measure of how natural the output looks. Lower is better.',
  },
  {
    name: 'BRISQUE',
    text: 'Blind/Referenceless Image Spatial Quality Evaluator — perceptual quality without a reference image. Lower is better.',
  },
  {
    name: 'Visibility',
    text: 'Estimated proportion of haze removed from the scene, derived from the dark channel. Higher is better.',
  },
];

export function AboutPage() {
  return (
    <div className="mx-auto max-w-3xl space-y-8">
      <header>
        <h1 className="text-3xl font-extrabold tracking-tight">
          About NightHaze
        </h1>
        <p className="mt-2 text-white/60">
          A hybrid deep learning and physics-based system for removing haze from
          nighttime photographs.
        </p>
      </header>

      <Card className="space-y-3">
        <h2 className="text-xl font-bold">The Problem</h2>
        <p className="text-sm leading-relaxed text-white/60">
          Daytime dehazing assumes a single, uniform source of illumination —
          the sun — and a globally constant atmospheric light. Night scenes
          break that assumption. Illumination comes from many artificial
          sources (street lamps, neon signs, headlights), each with its own
          colour and intensity, so atmospheric light varies from region to
          region rather than being global.
        </p>
        <p className="text-sm leading-relaxed text-white/60">
          Those light sources also scatter into visible glow halos, which
          classical methods misread as scene content. The Dark Channel Prior,
          the workhorse of daytime dehazing, fails outright: it assumes most
          local patches contain a near-zero-intensity channel, which is untrue
          around bright artificial lights. Night images are additionally
          noisier, because sensors run at high ISO in low light, and naive
          contrast enhancement amplifies that noise along with the signal.
        </p>
      </Card>

      <Card className="space-y-3">
        <h2 className="text-xl font-bold">Our Approach</h2>
        <p className="text-sm text-white/60">
          Six stages, combining a learned model with an explicit physical model:
        </p>
        <ol className="space-y-2">
          {PIPELINE_STAGES.map((stage, index) => (
            <li key={stage.name} className="flex gap-3 text-sm">
              <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-primary/20 text-xs font-semibold text-primary">
                {index + 1}
              </span>
              <span>
                <span className="font-semibold">{stage.name}</span>
                <span className="text-white/60"> — {stage.text}</span>
              </span>
            </li>
          ))}
        </ol>
      </Card>

      <Card className="space-y-3">
        <h2 className="text-xl font-bold">The Model</h2>
        <p className="text-sm leading-relaxed text-white/60">
          The deep learning core is FFA-Net (Feature Fusion Attention Network),
          an end-to-end dehazing architecture. It combines channel attention —
          which learns that haze affects colour channels unevenly — with pixel
          attention, which learns that haze is spatially non-uniform. Its
          feature-fusion structure weights information from three groups of
          nineteen residual attention blocks, letting the network preserve fine
          detail in thin-haze regions while working harder on dense ones. We use
          the pretrained weights released by the authors (trained on the RESIDE
          Indoor Training Set) and refine the output with our physics stage.
        </p>
        <blockquote className="border-l-2 border-primary pl-4 text-sm italic text-white/60">
          Qin, X., Wang, Z., Bai, Y., Xie, X., &amp; Jia, H. (2020). FFA-Net:
          Feature Fusion Attention Network for Single Image Dehazing.
          Proceedings of the AAAI Conference on Artificial Intelligence, 34(07),
          11908-11915.
        </blockquote>
      </Card>

      <Card className="space-y-3">
        <h2 className="text-xl font-bold">Quality Metrics</h2>
        <dl className="space-y-3">
          {METRICS.map((metric) => (
            <div key={metric.name} className="text-sm">
              <dt className="font-semibold">{metric.name}</dt>
              <dd className="text-white/60">{metric.text}</dd>
            </div>
          ))}
        </dl>
      </Card>

      <Card className="space-y-1 text-center">
        <p className="font-semibold">Final Year B.Tech CSE Project</p>
        <p className="text-sm text-white/40">
          Team, college, and batch details to be added.
        </p>
      </Card>
    </div>
  );
}
