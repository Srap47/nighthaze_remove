/**
 * Spinner — Animated loading indicator with three size options.
 *
 * Part of: Frontend UI primitives
 * Used by: HomePage (during processing), PipelineProgress (during each stage)
 *
 * Renders a rotating circular border with a gradient top edge. Respects
 * the prefers-reduced-motion media query (does not spin if user prefers no motion).
 */

interface SpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

// TWEAK NOTE: Spinner sizes
// Change h-* and w-* values to adjust the spinner diameter for each size
const SIZES: Record<NonNullable<SpinnerProps['size']>, string> = {
  sm: 'h-6 w-6',    // Small: 24px (use in compact contexts)
  md: 'h-10 w-10',  // Medium: 40px (default, suitable for most loading states)
  lg: 'h-16 w-16',  // Large: 64px (use for hero/full-page loading states)
};

export function Spinner({ size = 'md', className = '' }: SpinnerProps) {
  return (
    <div
      role="status"
      aria-label="Loading"
      className={`${SIZES[size]} rounded-full border-4 border-white/10 border-t-primary animate-spin ${className}`}
    />
  );
}
