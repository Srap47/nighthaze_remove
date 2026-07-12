/**
 * PipelineProgress — Animated stepper showing pipeline stages during processing.
 *
 * Part of: Frontend components
 * Used by: HomePage (displayed during "processing" state)
 *
 * Shows all six pipeline stages as dots that pulse in sequence with a staggered
 * animation. Since the backend processes requests synchronously (no per-stage
 * streaming), this is a visual indicator that work is happening, not a
 * real progress tracker. Actual timing details appear in MetricsDisplay after.
 */

// The six stages of the dehazing pipeline, in order:
// 1. Preprocess: Validate, denoise, gamma lift
// 2. Glow Detect: Detect bright light sources and halos
// 3. FFA-Net: Run the deep learning model
// 4. Recovery: Physics-based radiance recovery with glow blending
// 5. Enhance: CLAHE, denoise, sharpen, saturation
// 6. Metrics: Compute quality metrics (PSNR, SSIM, etc.)
const STAGES = [
  'Preprocess',
  'Glow Detect',
  'FFA-Net',
  'Recovery',
  'Enhance',
  'Metrics',
];

export function PipelineProgress() {
  return (
    <div className="w-full max-w-2xl">
      <div className="flex items-start justify-between gap-2">
        {STAGES.map((stage, index) => (
          <div key={stage} className="flex flex-1 flex-col items-center gap-2">
            {/* Pulsing dot with staggered animation delay so dots pulse in sequence */}
            <span
              className="h-2.5 w-2.5 animate-pulse rounded-full bg-primary"
              style={{ animationDelay: `${index * 120}ms` }}
            />
            <span className="text-center text-[11px] leading-tight text-white/50">
              {stage}
            </span>
          </div>
        ))}
      </div>

      {/* TWEAK NOTE: Processing time estimate and animation stagger
          - animationDelay: `${index * 120}ms` creates the staggered wave effect.
            Increase (e.g., 150ms) for slower wave, decrease (e.g., 80ms) for faster.
          - Text below is just a user expectation-setter. Update if typical times change
            (e.g., if GPUs become available, reduce the "~10–15 seconds" estimate).
      */}
      <p className="mt-5 text-center text-xs text-white/40">
        FFA-Net inference on CPU takes ~10–15 seconds
      </p>
    </div>
  );
}
