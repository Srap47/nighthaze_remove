// Static animated stepper shown while the backend runs the pipeline.
// The backend does not stream per-stage progress, so every dot pulses.

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

      <p className="mt-5 text-center text-xs text-white/40">
        FFA-Net inference on CPU takes ~10–15 seconds
      </p>
    </div>
  );
}
